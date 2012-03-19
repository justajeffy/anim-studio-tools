#ifndef _NAPALM_DELIGHT_SAMPLESX__H_
#define _NAPALM_DELIGHT_SAMPLESX__H_

#include <napalm/core/Table.h>
#include <SxWrappers.h>

namespace napalm_delight
{

//-------------------------------------------------------------------------------------------------------------
class SampleSx
{
public:
	//! construct from a napalm table in standard form
	SampleSx( napalm::c_object_table_ptr i_userParams, ::SxContext i_parent = 0 );
	~SampleSx();

	//! evaluate the shader on some samples
	bool sample(	const napalm::ObjectTable& i_params, //!< the surface parameter inputs (eg P, s, t etc)
					const napalm::ObjectTable& i_aovs,   //!< the aovs you want to sample (eg "Ci" etc)
					napalm::ObjectTable& o_data ) const; //!< the output buffers (eg V3f/float buffers)

	//! sample, but use aovs from the aovs entry in m_userParams
	bool sample( 	const napalm::ObjectTable& i_params, //!< the surface parameter inputs (eg P, s, t etc)
					napalm::ObjectTable& o_data ) const; //!< the output buffers (eg V3f/float buffers)

	//! dump some info about the shader
	void shaderInfo() const;

	//! the shader name
	std::string name() const;

	//! expose the context
	::SxContext getContext() { return *m_sx->get_data(); }

protected:
	napalm::c_object_table_ptr m_userParams;
	drd::SxContext m_sx;
	drd::SxShader m_shader;
	std::string m_shaderName; // for convenience
};

}

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
