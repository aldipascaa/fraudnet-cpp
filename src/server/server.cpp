#include "server.hpp"

#include <filesystem>
#include <optional>
#include <stdexcept>

#include <httplib.h>

#include "api.hpp"
#include "ui_assets.hpp"

namespace fraudnet {

namespace {
void send(httplib::Response& res, const ApiResult& r) {
  res.status = r.status;
  res.set_content(r.body.dump(), "application/json; charset=utf-8");
}

bool isJsonContentType(const std::string& ct) {
  // "application/json" optionally followed by parameters such as "; charset=utf-8"
  return ct.rfind("application/json", 0) == 0;
}

// Runs `fn` with the optional `?threshold=` override, or answers 400 when it is malformed.
template <class Fn>
ApiResult withThreshold(const httplib::Request& req, Fn&& fn) {
  if (!req.has_param("threshold")) return fn(std::nullopt);
  std::string error;
  const auto thr = parseThreshold(req.get_param_value("threshold"), error);
  if (!thr) return errorResult(400, "invalid_threshold", error);
  return fn(thr);
}
}  // namespace

ApiServer::ApiServer(Predictor predictor, ServerOptions options)
    : predictor_(std::move(predictor)), options_(std::move(options)), server_(std::make_unique<httplib::Server>()) {
  auto& svr = *server_;
  svr.set_payload_max_length(options_.maxBodyBytes);
  svr.set_read_timeout(10, 0);
  svr.set_write_timeout(10, 0);

  svr.set_post_routing_handler([](const httplib::Request&, httplib::Response& res) {
    res.set_header("X-Content-Type-Options", "nosniff");
    res.set_header("Referrer-Policy", "no-referrer");
    res.set_header("X-Frame-Options", "DENY");
    res.set_header("Content-Security-Policy",
                   "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'");
    if (res.get_header_value("Cache-Control").empty()) res.set_header("Cache-Control", "no-store");
  });

  // Every endpoint is registered under /api/v1 (canonical) and /api (deprecated alias).
  auto route = [&](const char* method, const std::string& suffix, std::size_t bodyLimit,
                   std::function<ApiResult(const Predictor&, const httplib::Request&)> fn) {
    for (const bool legacy : {false, true}) {
      const std::string path = (legacy ? "/api" : "/api/v1") + suffix;
      auto handler = [this, legacy, bodyLimit, fn, path](const httplib::Request& req, httplib::Response& res) {
        if (legacy) {
          res.set_header("Deprecation", "true");
          res.set_header("Link", "</api/v1" + req.path.substr(4) + ">; rel=\"successor-version\"");
        }
        if (req.method == "POST") {
          if (!isJsonContentType(req.get_header_value("Content-Type"))) {
            send(res, errorResult(415, "unsupported_media_type", "Content-Type must be application/json"));
            return;
          }
          if (req.body.size() > bodyLimit) {
            send(res, errorResult(413, "payload_too_large",
                                  "Request body exceeds " + std::to_string(bodyLimit) + " bytes for this endpoint"));
            return;
          }
        }
        send(res, fn(predictor_, req));
      };
      if (std::string(method) == "GET") svr.Get(path, handler);
      else svr.Post(path, handler);
    }
  };
  route("GET", "/health", 0, [](const Predictor& p, const httplib::Request&) { return handleHealth(p); });
  route("GET", "/model", 0, [](const Predictor& p, const httplib::Request&) { return handleModelInfo(p); });
  route("POST", "/predict", options_.maxSingleBodyBytes,
        [](const Predictor& p, const httplib::Request& r) {
          return withThreshold(r, [&](std::optional<double> t) { return handlePredict(p, r.body, t); });
        });
  route("POST", "/predict/batch", options_.maxBodyBytes,
        [](const Predictor& p, const httplib::Request& r) {
          return withThreshold(r, [&](std::optional<double> t) { return handleBatch(p, r.body, t); });
        });

  svr.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
    if (!res.body.empty()) return;  // handler already produced a body
    if (req.path.rfind("/api/", 0) == 0) {
      const int code = res.status;
      const char* name = code == 404   ? "not_found"
                         : code == 413 ? "payload_too_large"
                         : code == 405 ? "method_not_allowed"
                                       : "http_error";
      send(res, errorResult(code, name, code == 404 ? "No such endpoint: " + req.method + " " + req.path : "Request failed"));
    }
  });
  svr.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr) {
    send(res, errorResult(500, "internal_error", "Unexpected server error"));
  });

  if (options_.serveUi) {
    if (!options_.uiOverrideDir.empty()) {
      if (!std::filesystem::is_directory(options_.uiOverrideDir))
        throw std::runtime_error("UI directory does not exist: " + options_.uiOverrideDir);
      svr.set_mount_point("/", options_.uiOverrideDir);
      svr.set_file_extension_and_mimetype_mapping("svg", "image/svg+xml");
      svr.set_file_request_handler([](const httplib::Request&, httplib::Response& res) {
        res.set_header("Cache-Control", "no-cache");
      });
    } else {
      svr.Get(R"(/.*)", [](const httplib::Request& req, httplib::Response& res) {
        const EmbeddedAsset* a = findEmbeddedAsset(req.path);
        if (!a || req.path.rfind("/api/", 0) == 0) {
          res.status = 404;
          if (req.path.rfind("/api/", 0) == 0) send(res, errorResult(404, "not_found", "No such endpoint: GET " + req.path));
          else res.set_content("Not found", "text/plain; charset=utf-8");
          return;
        }
        res.set_header("Cache-Control", "no-cache");
        res.set_content(reinterpret_cast<const char*>(a->data), a->size, a->mime);
      });
    }
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
