#ifndef _NAPALM_TYPES_TYPEENABLE__H_
#define _NAPALM_TYPES_TYPEENABLE__H_

/*
 * @brief This file enables/disables napalm types at compile time. Comment/uncomment the
 * desired entries below. Definitions in earlier categories are coarser grain and take
 * precedence over the finer-grain control in later categories.
 */


/*
 * Category 1 - scalars
 */
#define ENABLE_NAPALM_TYPES_BASED_ON_INT		// V2i, V3i, B2i etc
#define ENABLE_NAPALM_TYPES_BASED_ON_FLOAT		// V2f, V3f, B2f etc
#define ENABLE_NAPALM_TYPES_BASED_ON_DOUBLE		// V2d, V3d, B2d etc
#define ENABLE_NAPALM_TYPES_BASED_ON_HALF		// V2h, V3h, B2h etc


/*
 * Category 2 - Imath types
 */
#define ENABLE_NAPALM_TYPES_IMATH				// V2h, B3f, M44d etc


/*
 * Category 3 - broad types
 */
#define ENABLE_NAPALM_TYPES_BOX					// B2f, B3i etc
#define ENABLE_NAPALM_TYPES_MATRIX				// M33f, M44d etc
#define ENABLE_NAPALM_TYPES_VEC					// V2i, V4f, V3h etc


/*
 * Category 4 - less-broad types
 */
#define ENABLE_NAPALM_TYPES_BOX2				// B2f, B2d etc
#define ENABLE_NAPALM_TYPES_BOX3				// B3f, B3d etc

#define ENABLE_NAPALM_TYPES_MATRIX33			// M33f, M33d etc
#define ENABLE_NAPALM_TYPES_MATRIX44			// M44f, M44d etc

#define ENABLE_NAPALM_TYPES_VEC2				// V2i, V2f etc
#define ENABLE_NAPALM_TYPES_VEC3				// V3i, V3f etc
#define ENABLE_NAPALM_TYPES_VEC4				// V4i, V4f etc


/*
 * Category 5 - specific types
 */
#define ENABLE_NAPALM_TYPES_BOOL
#define ENABLE_NAPALM_TYPES_TUPLE

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
