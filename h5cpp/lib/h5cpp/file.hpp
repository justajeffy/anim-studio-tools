#ifndef _H5CPP_FILE__H_
#define _H5CPP_FILE__H_

#include "hid_adaptor.hpp"

namespace h5cpp { namespace file {


	/*
	 * file access properties
	 */
	class file_access_props
	{
	public:

		file_access_props()
		:	m_fclose_degree_set(false)
		{}

		void set_fclose_degree(H5F_close_degree_t val)		{ m_fclose_degree=val; }

		void apply(hid_proplist proplist) const;

	protected:
		bool m_fclose_degree_set;

		H5F_close_degree_t m_fclose_degree;
	};


	/*
	 * @brief open
	 * See H5Fopen
	 * @param readonly If true, opens the file in read-only mode, otherwise opens for read-write.
	 */
	shared_hid_file open(const std::string& filepath, bool readonly = true,
		const file_access_props& aprops = file_access_props());


	/*
	 * @brief create
	 * see H5Fcreate
	 */
	shared_hid_file create(const std::string& filepath,
		const file_access_props& aprops = file_access_props());

	/*
	 * @brief given an hdf5 file, append a list of all open objects that match the given
	 * type to the given container. See H5Fget_obj_ids.
	 * @return the number of objects appended.
	 * @note Container type could be (eg): std::vector<hid_file>, std::list<hid_t>
	 */
	template<typename Container>
	unsigned int get_open_objects(const hid_file_adaptor& file, Container& obj_list,
		unsigned int types = H5F_OBJ_ALL);


///////////////////////// impl

template<typename Container>
unsigned int get_open_objects(const hid_file_adaptor& file, Container& obj_list, unsigned int types)
{
	typedef typename Container::value_type value_type;

	ssize_t nObjs = H5CPP_ERR_ON_NEG(H5Fget_obj_count(file.id(), types));
	if(nObjs == 0)
		return 0;

	hid_t* obj_ids_ = new hid_t[nObjs];
	H5CPP_ERR_ON_NEG(H5Fget_obj_ids(file.id(), types, nObjs, obj_ids_));

	std::back_insert_iterator<Container> bi = std::back_inserter(obj_list);
	for(ssize_t i=0; i<nObjs; ++i)
		bi = value_type(obj_ids_[i]);

	delete[] obj_ids_;

	return nObjs;
}

} } // ns


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
