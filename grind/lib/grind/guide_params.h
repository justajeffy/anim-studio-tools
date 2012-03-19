/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: guide_params.h 42544 2010-08-17 04:31:03Z allan.johns $"
 */

#ifndef grind_guide_params_h
#define grind_guide_params_h

//! @cond DEV

namespace grind {

//-------------------------------------------------------------------------------------------------
//! the parameters that describe how a vertex-guide will react in a simulation
struct GuideParams
{
	//! mass of strand segment
	float mass;
	//! stiffness at root of strand
	float stiffness_root;
	//! stiffness at tip of strand
	float stiffness_tip;
	//! gamma to control stiffness in between root and tip
	float stiffness_gamma;
};

} // grind

//! @endcond

#endif /* grind_guide_params_h */


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
