/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: utils.h 88551 2011-06-24 07:55:38Z luke.emrose $"
 */

#ifndef grind_utils_h
#define grind_utils_h

#include <string>

namespace grind
{
	bool hasGPU();
	bool hasEmulation();
	bool hasOpenGL();
	bool hasRX();

	// get an rx attribute
	bool grindGetRiAttribute( std::string const& attribute_name, std::string& result );

	// get the name of an object if we are in the renderman context
	std::string grindGetRiObjectName();
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
