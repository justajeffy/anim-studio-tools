/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: mesh_subdivide_types.h 42544 2010-08-17 04:31:03Z allan.johns $"
 */

#ifndef grind_mesh_subdivide_types_h
#define grind_mesh_subdivide_types_h

#include <iostream>

namespace grind {
namespace subd {

enum BorderSubdType{  BORDER_SUBD_NONE			//!< will leave hard borders as per src mesh
					, BORDER_SUBD_UP_TO_EDGE	//!< will smooth right up to edges (ie smooth mesh won't pull away from edges)
					};

struct Face
{
	int v0, v1, v2, v3;

	Face()
	: v0(-1), v1(-1), v2(-1), v3(-1)
	{}

	// allow dumping to a stream
	friend std::ostream &operator<<( std::ostream &stream, const Face& ob)
	{
	  stream << '(' << ob.v0 << ',' << ob.v1 << ',' << ob.v2 << ',' << ob.v3 << ')';
	  return stream;
	}
};


struct Edge
{
	//! vert indices of ends of edge
	int v0, v1;

	//! connected face indices
	int f0, f1;

	//! index of edge in face
	char f0_id, f1_id;

	Edge()
	: v0(-1), v1(-1), f0(-1), f1(-1), f0_id(-1), f1_id(-1)
	{}

	//! equivalence operator used to construct unique edge list (ie only comparing vert ids)
	bool operator==( const Edge& b ) const
	{
		// only compare vert ids
		return v0 == b.v0 && v1 == b.v1;
	}

	// allow dumping to a stream
	friend std::ostream &operator<<( std::ostream &stream, const Edge& ob)
	{
	  stream << '(' << ob.v0 << ',' << ob.v1 << ' ' << ob.f0 << ',' << ob.f1 << ' ' << int(ob.f0_id) << ',' << int(ob.f1_id) << ')';
	  return stream;
	}

};

} // subd namespace
} // grind namespace

#endif /* grind_mesh_subdivide_types_h */


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
