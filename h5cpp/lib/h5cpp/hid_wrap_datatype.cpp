#include "hid_wrap_datatype.h"
#include "datatype.hpp"

namespace h5cpp {


shared_hid_datatype::shared_hid_datatype(const hid_datatype& hid)
:	m_hid_holder(new hid_holder(hid))
{
}


shared_hid_datatype::shared_hid_datatype(const shared_hid_datatype& rhs)
:	m_hid_holder(rhs.m_hid_holder),
	m_cdtype(rhs.m_cdtype)
{
}


shared_hid_datatype::shared_hid_datatype(boost::shared_ptr<compound_datatype_base> cdtype)
:	m_cdtype(cdtype)
{
}


const hid_datatype& shared_hid_datatype::hid() const
{
	return (m_hid_holder)? m_hid_holder->m_hid : m_cdtype->hid();
}


hid_t shared_hid_datatype::id() const
{
	return (m_hid_holder)? m_hid_holder->m_hid.id() : m_cdtype->hid().id();
}


shared_hid_datatype::operator bool() const
{
	return bool(id());
}


void shared_hid_datatype::reset(const hid_datatype& hid)
{
	m_hid_holder.reset(new hid_holder(hid));
}


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
