#include "server.hpp"

#include <filesystem>
#include <stdexcept>

#include <httplib.h>

#include "api.hpp"

namespace fraudnet {

namespace {
void send(httplib::Response& res, const ApiResult& r) {
  res.status = r.status;
  res.set_content(r.body.dump(), "application/json; charset=utf-8");
}
}  // namespace

ApiServer::ApiServer(FraudModel model, std::string uiDir)
    : model_(std::move(model)), uiDir_(std::move(uiDir)), server_(std::make_unique<httplib::Server>()) {
  auto& svr = *server_;
  svr.set_payload_max_length(8 * 1024 * 1024);

  svr.set_post_routing_handler([](const httplib::Request&, httplib::Response& res) {
    res.set_header("X-Content-Type-Options", "nosniff");
    res.set_header("Referrer-Policy", "no-referrer");
    res.set_header("X-Frame-Options", "DENY");
    res.set_header("Content-Security-Policy",
                   "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'");
    if (res.get_header_value("Cache-Control").empty()) res.set_header("Cache-Control", "no-store");
  });

  svr.Get("/api/health", [this](const httplib::Request&, httplib::Response& res) { send(res, handleHealth(model_)); });
  svr.Get("/api/model", [this](const httplib::Request&, httplib::Response& res) { send(res, handleModelInfo(model_)); });
  svr.Post("/api/predict", [this](const httplib::Request& req, httplib::Response& res) {
    send(res, handlePredict(model_, req.body));
  });
  svr.Post("/api/predict/batch", [this](const httplib::Request& req, httplib::Response& res) {
    send(res, handleBatch(model_, req.body));
  });

  svr.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
    if (!res.body.empty()) return;  // handler already produced a JSON error
    if (req.path.rfind("/api/", 0) == 0) {
      const int code = res.status;
      const char* name = code == 404 ? "not_found" : (code == 413 ? "payload_too_large" : "http_error");
      send(res, errorResult(code, name, code == 404 ? "No such endpoint: " + req.method + " " + req.path : "Request failed"));
    }
  });
  svr.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr) {
    send(res, errorResult(500, "internal_error", "Unexpected server error"));
  });

  if (!uiDir_.empty() && std::filesystem::is_directory(uiDir_)) {
    svr.set_mount_point("/", uiDir_);
    svr.set_file_extension_and_mimetype_mapping("svg", "image/svg+xml");
    svr.set_file_request_handler([](const httplib::Request&, httplib::Response& res) {
      res.set_header("Cache-Control", "no-cache");
    });
  }
}

ApiServer::~ApiServer() { stop(); }

int ApiServer::bind(const std::string& host, int port) {
  const int p = port == 0 ? server_->bind_to_any_port(host) : (server_->bind_to_port(host, port) ? port : -1);
  if (p < 0) throw std::runtime_error("cannot bind to " + host + ":" + std::to_string(port) + " (port in use?)");
  return p;
}

void ApiServer::run() { server_->listen_after_bind(); }

void ApiServer::start() {
  thread_ = std::thread([this] { server_->listen_after_bind(); });
  while (!server_->is_running()) std::this_thread::yield();
}

void ApiServer::stop() {
  if (server_) server_->stop();
  if (thread_.joinable()) thread_.join();
}

}  // namespace fraudnet
