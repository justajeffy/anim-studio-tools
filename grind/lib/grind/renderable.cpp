/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: renderable.cpp 42544 2010-08-17 04:31:03Z allan.johns $"
 */

//-------------------------------------------------------------------------------------------------
#include "renderable.h"
#include "context.h"
#include "bbox.h"

#include <stdexcept>

//-------------------------------------------------------------------------------------------------
using namespace grind;

//-------------------------------------------------------------------------------------------------
Renderable::Renderable()
{
}

//-------------------------------------------------------------------------------------------------
Renderable::~Renderable()
{
}

//-------------------------------------------------------------------------------------------------
void Renderable::render( float lod ) const
{
	try
	{
		if ( ContextInfo::instance().hasOpenGL() )
		{
			dumpGL( lod );
		}
		else if ( ContextInfo::instance().hasRX() )
		{
			dumpRib( lod );
		}
		else
		{
			throw std::runtime_error( "can't render without OpenGL or RX render context" );
		}
	}
	catch ( std::exception& e )
	{
		throw std::runtime_error( std::string( "Render Error: " ) + e.what() );
	}
}

//-------------------------------------------------------------------------------------------------
void Renderable::dumpRib( float lod ) const
{
	throw std::runtime_error( "object doesn't support Rib rendering" );
}

//-------------------------------------------------------------------------------------------------
void Renderable::dumpGL( float lod ) const
{
	throw std::runtime_error( "object doesn't support OpenGL rendering" );
}

//-------------------------------------------------------------------------------------------------
BBox Renderable::getBounds() const
{
	throw std::runtime_error( "getBounds() not yet supported for object" );
}


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
