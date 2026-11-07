#include "ui_assets.hpp"

namespace fraudnet {

const EmbeddedAsset* findEmbeddedAsset(std::string_view urlPath) {
  if (urlPath == "/") urlPath = "/index.html";
  std::size_t n = 0;
  const EmbeddedAsset* all = embeddedUiAssets(n);
  for (std::size_t i = 0; i < n; ++i)
    if (urlPath == all[i].path) return &all[i];
  return nullptr;
}

}  // namespace fraudnet
