/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/Renderer.h $"
 * SVN_META_ID = "$Id: Renderer.h 68962 2011-02-25 00:22:22Z stephane.bertout $"
 */

#ifndef bee_Renderer_h
#define bee_Renderer_h
#pragma once

#include "../kernel/types.h"
#include "../kernel/color.h"
#include "../kernel/smartPointers.h"
#include <GL/glew.h>
#include <GL/glut.h>

namespace bee
{
	class Texture;

	//-------------------------------------------------------------------------------------------------
	//! Renderer is a GL utility class encapsulating some GL rendering functions
	class Renderer
	{
	public:
		//! Different CullFace mode available
		enum CullFace
		{
			eFront = 0,
			eBack,
			eFrontAndBack,
		};

		//! Different ClearFlag mode available
		enum ClearFlag
		{
			eColor = 0x1,
			eDepth = 0x2,
			eStencil = 0x4,
		};

		//! Default constructor
		Renderer();
		//! Destructor
		~Renderer() {}

		//! Perform some basic GL init
		static void init();

		//! Init GL ext only (via glew)
		static void init_extensions();

		//! Set the Clear Colour to use
		void setClearColour( const Colour & );
		//! Clear the current frame buffer
		void clear( UInt a_ClearFlags );

		//! Enable Face Culling
		void enableCullFace( CullFace a_CullFace );
		//! Disable Face Culling
		void disableCullFace();

		//! Enable Wireframe
		void enableWireFrame( bool );

		std::string initLut();

		const SharedPtr<Texture> & getLutTexture() const
		{
			return m_Lut3dTexture;
		}

	private:
		Colour m_ClearColor;

		bool m_CullFaceEnabled;
		CullFace m_CullFace;

		SharedPtr<Texture> m_Lut3dTexture;
	};
}
#endif // bee_Renderer_h


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
