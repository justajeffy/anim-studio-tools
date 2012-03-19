

#include "napalmDelight/exceptions.h"
#include "napalmDelight/render.h"

using namespace napalm_delight;

NapalmDelightError::NapalmDelightError( const std::string& msg )
: NapalmError( napalm_delight::renderManObjectName() + ": " + msg )
{}
