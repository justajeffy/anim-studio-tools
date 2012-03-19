/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: bbox.h 102157 2011-09-12 01:45:39Z stephane.bertout $"
 */

#ifndef grind_bbox_h
#define grind_bbox_h

//-------------------------------------------------------------------------------------------------

#include "renderable.h"
#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathBox.h>
#include <vector>
#include <oriented_bounding_box/OrientedBoundingBox.h>

namespace grind
{
//-------------------------------------------------------------------------------------------------
//! a bounding box (an Oriented Bounding Box that is renderable)
class BBox
: public OrientedBoundingBox<float>
, public Renderable
{
public:
	typedef Imath::V3f VecType;
	typedef VecType::BaseType BaseType;
	typedef Imath::Box< Imath::V3f > BoxType;
	typedef Imath::M44f MatType;

	//! default constructor (empty box)
	BBox();

	template< typename Iter >
	BBox( Iter i, Iter e )
		: m_ColourIndex( 0 ),
		  m_Colour( 0 )
	{
		populate( i, e );
	}

	BBox( VecType min, VecType max )
		: m_ColourIndex( 0 ),
		  m_Colour( 0 )
	{
		GetBox().min = min;
		GetBox().max = max;
	}

	bool operator==(const BBox & a_Other ) const
	{
		return GetBox() == a_Other.GetBox() &&
				GetRotation() == a_Other.GetRotation();
	}

	template< typename Iter >
	void populate( Iter i, Iter e, int mode = -1 )
	{
		switch( ( mode >= 0 ) ? mode : s_BBOX_MODE )
		{
			case 2:
				FitBoxCovariance( i, e );
				break;
			case 1:
				FitBox( i, e );
				break;
			default:
				FitBoxNoRotation( i, e );
				break;
		}
	}

	void setColourIndex( int id ) { m_ColourIndex = id; }
	//! make the box render with a certain colour
	void setColour( unsigned int col ) { m_Colour = col; }
	//! expand the bounding box by a scalar amount in each axis
	void pad( BaseType v )
	{
		VecType & min = GetBox().min;
		VecType & max = GetBox().max;
		min.x -= v; min.y -= v; min.z -= v;
		max.x += v; max.y += v; max.z += v;
	}
	bool isEmpty() const
	{
		return GetBox().isEmpty();
	}
	VecType center() const
	{
		VecType v;
		GetTransform().multVecMatrix(GetBox().center(),v);
		return v;
	}
private:
	void dumpGL( float lod ) const;
	void dumpRib( float lod ) const;
	void buildArrays( std::vector< VecType >& result ) const;
	int m_ColourIndex;
	unsigned int m_Colour;
	static int s_BBOX_MODE;
};

}
#endif /* grind_bbox_h */


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
