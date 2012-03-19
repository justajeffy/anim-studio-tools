/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/kernel/pythonHelper.h $"
 * SVN_META_ID = "$Id: pythonHelper.h 17302 2009-11-18 06:20:42Z david.morris $"
 */

#ifndef bee_pythonHelper_h
#define bee_pythonHelper_h
#pragma once

namespace bee
{
	#define ADD_PY_PROPERTY( type, name )								\
			.add_property( #name, &type::get##name, &type::set##name )

	#define ADD_PY_PROPERTY_RDONLY( type, name )						\
			.add_property( #name, &type::get##name )

	#define DEF_PY_FUNCTION( type, name )								\
			.def( #name, &type::name )

	#define DEF_PY_VALUE( type, name )								\
		.value( #name, type::name )

	#define DEF_PY_GLOBAL_FUNCTION( name )							\
			def( #name, &name )
}

#endif // bee_pythonHelper_h


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
