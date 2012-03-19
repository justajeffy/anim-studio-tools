#ifndef _NAPALM_DELIGHT_BUFFER_HELPERS__H_
#define _NAPALM_DELIGHT_BUFFER_HELPERS__H_

//-------------------------------------------------------------------------------------------------
#define SET_UP_BUFFER( TABLE, NAME, TYPE, SZ ) \
napalm::TYPE ## Ptr TABLE ## _ ## NAME ## _buf_ptr( new napalm::TYPE( SZ ) ); \
(TABLE).setEntry( #NAME, TABLE ## _ ## NAME ## _buf_ptr );

//-------------------------------------------------------------------------------------------------
#define SET_UP_BUFFER_FR( TABLE, NAME, TYPE, SZ ) \
SET_UP_BUFFER( TABLE, NAME, TYPE, SZ ) \
napalm::TYPE::w_type TABLE ## _ ## NAME ## _fr = TABLE ## _ ##  NAME ## _buf_ptr->rw(); \
napalm::TYPE::w_type::iterator TABLE ## _ ## NAME ## _iter = TABLE ## _ ## NAME ## _fr.begin();

#endif


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
