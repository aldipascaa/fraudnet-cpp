#pragma once
#include <cstddef>
#include <string_view>

namespace fraudnet {

/// One static web-UI file compiled into the binary (generated at build time from ui/ by
/// cmake/embed_ui.cmake into <build>/generated/ui_assets.cpp).
struct EmbeddedAsset {
  const char* path;  // URL path, e.g. "/index.html"
  const char* mime;
  const unsigned char* data;
  std::size_t size;
};

/// All embedded UI files.
const EmbeddedAsset* embeddedUiAssets(std::size_t& count);
/// Exact-match lookup by URL path ("/" maps to "/index.html"); nullptr if absent.
const EmbeddedAsset* findEmbeddedAsset(std::string_view urlPath);

}  // namespace fraudnet
