#ifndef _NAPALM_DELIGHT_RENDER__H_
#define _NAPALM_DELIGHT_RENDER__H_

#include <napalm/core/Table.h>

namespace napalm_delight
{

//! is there an active renderman context?
bool hasRenderManContext();

//! if we are in renderman then return the value of the identified name attached to this object
std::string renderManObjectName();

//! Napalm binding for RiSphereV()
void sphere( const napalm::ObjectTable& o );

//! Napalm binding for RiPointsV()
void points( const napalm::ObjectTable& o );

//! Napalm binding for RiCurvesV()
void curves( const napalm::ObjectTable& o );

//! Napalm binding for archives
void archives( const napalm::ObjectTable& o );

}

#endif


/***
    Copyright 2008-2012 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios)

    This file is part of anim-studio-tools.

    anim-studio-tools is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    anim-studio-tools is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with anim-studio-tools.  If not, see <http://www.gnu.org/licenses/>.
***/
