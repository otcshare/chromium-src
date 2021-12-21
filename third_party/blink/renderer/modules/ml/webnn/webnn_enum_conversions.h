#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_ENUM_CONVERSIONS_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_ENUM_CONVERSIONS_H_

#include <webnn/webnn.h>

namespace WTF {
class String;
}

namespace blink {

// Convert ml bitfield values to Webnn enums. These have the same value.
template <typename WebnnEnum>
WebnnEnum AsWebnnEnum(uint32_t ml_enum) {
  return static_cast<WebnnEnum>(ml_enum);
}

// Convert WebNN string enums to Webnn enums.
template <typename WebnnEnum>
WebnnEnum AsWebnnEnum(const WTF::String& ml_enum);

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_ENUM_CONVERSIONS_H_
