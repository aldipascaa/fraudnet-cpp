#pragma once
#include <memory>
#include <string>
#include <thread>

#include "model.hpp"

namespace httplib {
class Server;
}

namespace fraudnet {

/// REST API + static UI on a loopback address.
class ApiServer {
 public:
  ApiServer(FraudModel model, std::string uiDir);
  ~ApiServer();
  ApiServer(const ApiServer&) = delete;
  ApiServer& operator=(const ApiServer&) = delete;

  /// Binds to host:port (port 0 = any free port) and returns the bound port; throws on failure.
  int bind(const std::string& host, int port);
  /// Blocks serving requests until stop() is called.
  void run();
  /// Serves in a background thread.
  void start();
  void stop();

 private:
  FraudModel model_;
  std::string uiDir_;
  std::unique_ptr<httplib::Server> server_;
  std::thread thread_;
};

}  // namespace fraudnet
