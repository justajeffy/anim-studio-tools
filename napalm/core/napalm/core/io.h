#ifndef _NAPALM_IO__H_
#define _NAPALM_IO__H_

#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include "typedefs.h"


namespace napalm {

	// supported file formats
	enum NapalmFileType
	{
		FILE_BOOST_UNKNOWN	= 0,	// treated as .nap
		FILE_BOOST_BINARY 	= 1,	// .nap
		FILE_BOOST_XML		= 2		// .xml
	};

	// save options
	struct SaveOptions
	{
		SaveOptions(unsigned int compression = 0)
		:	m_compression(compression){}

		// if zero, no data compression will occur. A higher number indicates a higher
		// level of compression, but this will mean different things for different archives.
		// Note that some archives do not support compression.
		unsigned int m_compression;
	};

	// load options
	struct LoadOptions
	{
		LoadOptions(bool delayLoad = true)
		:	m_delayLoad(delayLoad){}

		// if true, buffers will put off loading their data from disk until the data is
		// first accessed. Note that this is only possible with certain archives.
		bool m_delayLoad;
	};


	// save a napalm object to disk
	void save(c_object_ptr obj, const std::string& filepath, const SaveOptions& op = SaveOptions());

	// save a napalm object to memory
	CharBufferPtr saveToMemory(c_object_ptr obj, const SaveOptions& op = SaveOptions());


	// load a napalm object from disk
	object_ptr load(const std::string& filepath, const LoadOptions& op = LoadOptions());

	// load a napalm object from memory
	object_ptr loadFromMemory(CharBufferCPtr buf, const LoadOptions& op = LoadOptions());


	// test two objects for equality. This is actually an io operation - both objects
	// are serialised into memory, then this memory is byte-compared to test for equality
	bool areEqual(c_object_ptr obj1, c_object_ptr obj2);


	namespace detail
	{

		struct NapalmFileHeader
		{
			NapalmFileHeader();
			bool valid() const;
			bool newerAPI() const;
			template<class Archive> void serialize(Archive& ar, const unsigned int version);

			int m_majorVersion;
			int m_minorVersion;
			int m_patchVersion;
			char m_N;
			char m_A;
			char m_P;
		};


		struct archive_tracking_scope;

		struct archive_info
		{
			std::string m_filepath;
			std::ifstream* m_fs;
			SaveOptions m_saveOp;
			LoadOptions m_loadOp;
		};

		// todo make threadsafe
		class archive_tracker
		{
			typedef std::map<const void*, archive_info> map_type;
			friend struct archive_tracking_scope;

		public:
			template<typename Archive>
			static const archive_info& getInfo(const Archive& ar)
			{
				const void* arp = static_cast<const void*>(&ar);
				map_type::iterator it = s_info_map.find(arp);
				assert(it != s_info_map.end());
				return it->second;
			}

		protected:
			static map_type s_info_map;
		};
	}

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
