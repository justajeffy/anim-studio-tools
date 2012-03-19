/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/py/lib/kernel/bindings.cpp $"
 * SVN_META_ID = "$Id: bindings.cpp 27186 2010-04-07 01:35:04Z david.morris $"
 */

#include <boost/python.hpp>
using namespace boost::python;

#include <kernel/pythonHelper.h>
#include <string>

using namespace bee;

namespace
{
	const char *
	getVersion()
	{
		return "beepy rev:$Revision: 27186 $ " __DATE__ " @ " __TIME__;
	}
}

void importKernelBindings()
{
	def( "getVersion", getVersion );
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
