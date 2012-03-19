#include "../TypedAttribBind.hpp"
#include "../TypedBufferBind.hpp"
#include "../SetBind.hpp"
#include "typelabels.h"


#define _NAPALM_TYPE_BIND_OP(T)										\
{																	\
	using namespace napalm;											\
	std::string label(type_label<T>::value());						\
	TypedAttribConvert<T>();										\
	TypedAttribConvertVector<T>();									\
	TypedAttribConvertSet<T>();										\
	TypedBufferBind<T>(label + "Buffer");							\
	TypedBufferBind<counted_vector<T>::type>(label + "VBuffer");	\
	SetBind<T>( label + "Set" );									\
}


// todo move labels outta here


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
