/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/kernel/smartPointers.h $"
 * SVN_META_ID = "$Id: smartPointers.h 17302 2009-11-18 06:20:42Z david.morris $"
 */

#ifndef bee_smartPointers_h
#define bee_smartPointers_h
#pragma once

#include <memory.h>
// use boost implementation for now
#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/intrusive_ptr.hpp>

namespace bee
{
	// help yourself...
	#define AutoPtr std::auto_ptr
	#define ScopedPtr boost::scoped_ptr
	#define ScopedArray boost::scoped_array
	#define SharedPtr boost::shared_ptr
	#define SharedArray boost::shared_array
	#define WeakPtr boost::weak_ptr
	#define IntrusivePtr boost::intrusive_ptr
}

#endif // bee_smartPointers_h



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
