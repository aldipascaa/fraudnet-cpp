#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <thread>

#include "predictor.hpp"

namespace httplib {
class Server;
}

namespace fraudnet {

struct ServerOptions {
  /// Hard cap on any request body; larger requests get 413 before they are parsed.
  std::size_t maxBodyBytes = 4 * 1024 * 1024;
  /// Cap for the single-transaction endpoint (a valid transaction is ~300 bytes).
  std::size_t maxSingleBodyBytes = 64 * 1024;
  /// Serve the web UI embedded in the binary.
  bool serveUi = true;
  /// If non-empty, serve the UI from this directory instead of the embedded copy (for UI development).
  std::string uiOverrideDir;
};

/// REST API (versioned under /api/v1, legacy /api/* aliases) + web UI on a loopback address.
/// The server only ever holds a `Predictor` (immutable, thread-safe), so concurrent requests
/// need no locking.
class ApiServer {
 public:
  explicit ApiServer(Predictor predictor, ServerOptions options = {});
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
  Predictor predictor_;
  ServerOptions options_;
  std::unique_ptr<httplib::Server> server_;
  std::thread thread_;
};

}  // namespace fraudnet
