/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/py/lib/gl/PyRenderTarget.h $"
 * SVN_META_ID = "$Id: PyRenderer.h 68962 2011-02-25 00:22:22Z stephane.bertout $"
 */

#ifndef bee_py_PyRenderTarget_h
#define bee_py_PyRenderTarget_h

#include <boost/shared_ptr.hpp>
#include <kernel/types.h>

//----------------------------------------------------------------------------
namespace bee
{
	//----------------------------------------------------------------------------
	class Texture;
	class RenderTarget;
	//----------------------------------------------------------------------------
	namespace py
	{
		class Texture;

		//----------------------------------------------------------------------------
		//! a render target python wrapper
		class RenderTarget
		{
		public:
			//! default constructor
			RenderTarget();

			//! constructor with specific dimensions
			RenderTarget( bee::UInt a_width, bee::UInt a_height );

			//! default destructor
			~RenderTarget();

			//! create a render target
			bool create( bee::UInt a_width, bee::UInt a_height );

			//! destroy the bee::RenderTarget
			bool destroy();

			//! use with a drawing context (also enables read)
			void use();

			//! un-use (when finished drawing)
			void release();

			//! to pass to a program?
			const Texture * getTexture( bee::UInt a_unit ) const;

			//! get width
			bee::UInt getWidth() const;

			//! get height
			bee::UInt getHeight() const;

		private:

			//! the underlying bee render target
			boost::shared_ptr< bee::RenderTarget > m_renderTarget;
		};

	} // namespace py
} // namespace bee

#endif // bee_py_PyRenderTarget_h


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
