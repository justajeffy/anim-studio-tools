#include "io.h"
#include "system.h"
#include "Table.h"
#include "exceptions.h"
#include "archives.h"
#include "TypedBuffer.h"
#include <pystring.h>
#include <utility>
#include <sstream>


using boost::serialization::make_nvp;
namespace ba = boost::archive;

namespace napalm {

namespace detail {

NapalmFileHeader::NapalmFileHeader()
:	m_N('N'), m_A('A'), m_P('P')
{
	m_majorVersion = NapalmSystem::instance().getMajorVersion();
	m_minorVersion = NapalmSystem::instance().getMinorVersion();
	m_patchVersion = NapalmSystem::instance().getPatchVersion();
}


bool NapalmFileHeader::valid() const
{
	return ((m_N == 'N') &&
			(m_A == 'A') &&
			(m_P == 'P') &&
			(m_majorVersion >= 0) &&
			(m_minorVersion >= 0) &&
			(m_patchVersion >= 0));
}

bool NapalmFileHeader::newerAPI() const
{
	return (std::pair<int,int>(NapalmSystem::instance().getMajorVersion(),
				NapalmSystem::instance().getMinorVersion()) <
				std::pair<int,int>(m_majorVersion, m_minorVersion));
}

template<class Archive>
void NapalmFileHeader::serialize(Archive& ar, const unsigned int version)
{
	ar & make_nvp("majorVersion", m_majorVersion);
	ar & make_nvp("minorVersion", m_minorVersion);
	ar & make_nvp("patchVersion", m_patchVersion);
	ar & make_nvp("N", m_N);
	ar & make_nvp("A", m_A);
	ar & make_nvp("P", m_P);
}


struct archive_tracking_scope
{
	template<typename Archive>
	archive_tracking_scope(const Archive& ar, archive_info& info)
	:	m_arp(static_cast<const void*>(&ar))
	{
		archive_tracker::s_info_map.insert(
			archive_tracker::map_type::value_type(m_arp, info));
	}

	~archive_tracking_scope() {
		archive_tracker::s_info_map.erase(m_arp);
	}

	const void* m_arp;
};


archive_tracker::map_type archive_tracker::s_info_map;

} // detail ns


NapalmFileType getNapalmFileType(const std::string& filepath)
{
	std::vector<std::string> strs;
	pystring::split(filepath, strs, ".");
	if(strs.size() < 2)
		return FILE_BOOST_UNKNOWN;

	const std::string& ext = strs.back();
	if(ext == "nap")
		return FILE_BOOST_BINARY;
	else if(ext == "xml")
		return FILE_BOOST_XML;

	return FILE_BOOST_UNKNOWN;
}


template<class Archive>
void serializeHeader(Archive& ar)
{
	detail::NapalmFileHeader hdr;
	ar & make_nvp("header", hdr);
}


template<class Archive>
void validateHeader(std::ifstream& fs, const std::string& filepath)
{
	detail::NapalmFileHeader hdr;
	fs.seekg(0);
	std::string errMsg("not a napalm file: " + filepath);

	try
	{
		Archive ar(fs);
		ar & make_nvp("header", hdr);
	}
	catch(ba::archive_exception& e)
	{
		if(e.code == ba::archive_exception::stream_error) {
			throw napalm::NapalmFileError(e.what());
		}
		else {
			throw napalm::NapalmSerializeError(errMsg);
		}
	}
	catch(std::exception& e) {
		throw napalm::NapalmSerializeError(errMsg);
	}

	if(!hdr.valid()) {
		throw napalm::NapalmSerializeError(errMsg);
	}

	if(hdr.newerAPI())
	{
		std::cerr << "File '" << filepath
			<< "' was written with a newer napalm API ("
			<< hdr.m_majorVersion << '.'
			<< hdr.m_minorVersion << '.'
			<< hdr.m_patchVersion << ") than this napalm API ("
			<< NapalmSystem::instance().getVersionString()
			<< "). Further data serialization may fail."
			<< std::endl;
	}
}


template<typename Archive>
void _save(object_ptr obj, const std::string& filepath, const SaveOptions& op)
{
	std::ofstream fs(filepath.c_str());
	Archive ar(fs);
	serializeHeader(ar);

	detail::archive_info ai;
	ai.m_saveOp = op;
	detail::archive_tracking_scope as(ar, ai);

	ar & make_nvp("root", obj);
}


template<typename Archive>
object_ptr _load(const std::string& filepath, const LoadOptions& op)
{
	std::ifstream fs(filepath.c_str());
	validateHeader<Archive>(fs, filepath);

	fs.seekg(0);
	Archive ar(fs);
	serializeHeader(ar);

	detail::archive_info ai;
	ai.m_loadOp = op;
	ai.m_fs = &fs;
	ai.m_filepath = filepath;
	detail::archive_tracking_scope as(ar, ai);

	object_ptr obj;
	try {
		ar & make_nvp("root", obj);
	}
	catch(ba::archive_exception& e) {
		throw napalm::NapalmSerializeError(e.what());
	}

	return obj;
}


void save(c_object_ptr obj, const std::string& filepath, const SaveOptions& op)
{
	// this step is extremely important. shared_ptr<T> and shared_ptr<const T> serialize
	// differently, mismatching constness on save/load usually causes a hard crash. We
	// assume that we never want to serialise a shared_ptr<const T>.
	object_ptr _obj = boost::const_pointer_cast<Object>(obj);

	std::ofstream fs(filepath.c_str());

	NapalmFileType ft = getNapalmFileType(filepath);
	switch(ft)
	{
	case FILE_BOOST_XML:
		_save<ba::xml_oarchive>(_obj, filepath, op);
		break;
	default:
		_save<ba::binary_oarchive>(_obj, filepath, op);
		break;
	}
}


object_ptr load(const std::string& filepath, const LoadOptions& op)
{
	object_ptr obj;
	std::ifstream fs(filepath.c_str());

	NapalmFileType ft = getNapalmFileType(filepath);
	switch(ft)
	{
	case FILE_BOOST_XML:
		obj = _load<ba::xml_iarchive>(filepath, op);
		break;
	default:
		obj = _load<ba::binary_iarchive>(filepath, op);
		break;
	}

	return obj;
}


CharBufferPtr saveToMemory(c_object_ptr obj, const SaveOptions& op)
{
	std::string data;
	{
		std::ostringstream strm;
		{
			ba::binary_oarchive ar(strm);
			detail::archive_info ai;
			ai.m_saveOp = op;
			detail::archive_tracking_scope as(ar, ai);

			// very important step, see earlier comment in save()
			object_ptr _obj = boost::const_pointer_cast<Object>(obj);
			ar & make_nvp("root", _obj);
		}
		data = strm.str();
	}

	CharBufferPtr buf(new CharBuffer(data.begin(), data.end()));
	return buf;
}


object_ptr loadFromMemory(CharBufferCPtr buf, const LoadOptions& op)
{
	assert(buf);
	LoadOptions op2 = op;
	op2.m_delayLoad = false;

	std::istringstream strm;
	{
		CharBuffer::r_type fr = buf->r();
		std::string data(fr.begin(), fr.end());
		strm.str(data);
	}

	ba::binary_iarchive ar(strm);
	detail::archive_info ai;
	ai.m_loadOp = op2;
	detail::archive_tracking_scope as(ar, ai);

	object_ptr obj;
	ar & make_nvp("root", obj);

	return obj;
}


// note that obj1 and obj2 will not have any delay-loaded child objects forced into memory.
bool areEqual(c_object_ptr obj1, c_object_ptr obj2)
{
	CharBufferPtr buf1 = saveToMemory(obj1);
	CharBufferPtr buf2 = saveToMemory(obj2);
	CharBuffer::r_type fr1 = buf1->r();
	CharBuffer::r_type fr2 = buf2->r();

	return ( (fr1.size() == fr2.size()) &&
		std::equal(fr1.begin(), fr1.end(), fr2.begin()) );
}


} // napalm ns






