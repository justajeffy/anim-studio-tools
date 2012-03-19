#ifndef _NAPALM_DELAYED_VECTOR__H_
#define _NAPALM_DELAYED_VECTOR__H_

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/mpl/bool.hpp>
#include "io.h"
#include "typedefs.h"


namespace napalm {

	namespace detail {

		struct DelayLoadData
		{
			std::string m_filepath;
			NapalmFileType m_fileType;
			std::size_t m_size;
			std::streampos m_fpos;
		};
	}

	/*
	 * @class delayed_vector
	 * @brief
	 * A wrapper class for counted_vector that is able to lazily load data from a boost
	 * archive only when necessary. Only some archives (eg binary_iarchive) support this.
	 */
	template<typename T>
	class delayed_vector
	{
	public:

		typedef typename counted_vector<T>::type vector_type;

		delayed_vector(std::size_t size, const T& value = T());

		delayed_vector(const delayed_vector& rhs);

		~delayed_vector();

		delayed_vector& operator=(const delayed_vector& rhs);

		// Return the number of elements in the vector (memory resident or not)
		std::size_t size() const;

		// Return the number of elements resident in memory
		std::size_t clientSize() const;

		// Return true if the contents have been loaded from disk, false otherwise
		bool isLoaded() const { return !m_delayData; }

		// Perform a potentially destructive resize of the vector
		void resize(std::size_t size, bool destructive = false);

		// destroy any extra allocated memory
		void shrink();

		// Data access - this may cause data to be loaded from disk
		vector_type& data();
		const vector_type& data() const;

	protected:

		void load() const;

		friend class boost::serialization::access;
		template<class Archive> void save(Archive& ar, const unsigned int version) const;
		template<class Archive> void save_(Archive& ar, const unsigned int version, boost::mpl::true_) const;
		template<class Archive> void save_(Archive& ar, const unsigned int version, boost::mpl::false_) const;
		template<class Archive> void load(Archive& ar, const unsigned int version);
		template<class Archive> void load_(Archive& ar, const unsigned int version, boost::mpl::true_);
		template<class Archive> void load_(Archive& ar, const unsigned int version, boost::mpl::false_);
		BOOST_SERIALIZATION_SPLIT_MEMBER()

		static void loadDelayData(boost::archive::binary_iarchive& ar, const detail::archive_info& ai,
			const unsigned int version, detail::DelayLoadData& dd, unsigned int numElems);

	protected:

		mutable vector_type m_data;

		mutable boost::shared_ptr<const detail::DelayLoadData> m_delayData;
	};


///////////////////////// impl

template<typename T>
delayed_vector<T>::delayed_vector(std::size_t size, const T& value)
:	m_data(size, value)
{
}


template<typename T>
delayed_vector<T>::delayed_vector(const delayed_vector& rhs)
:	m_data(rhs.m_data),
	m_delayData(rhs.m_delayData)
{
}


template<typename T>
delayed_vector<T>::~delayed_vector()
{
}


template<typename T>
delayed_vector<T>& delayed_vector<T>::operator=(const delayed_vector& rhs)
{
	m_data = rhs.m_data;
	m_delayData = rhs.m_delayData;
	return *this;
}


template<typename T>
typename delayed_vector<T>::vector_type& delayed_vector<T>::data()
{
	load();
	return m_data;
}


template<typename T>
const typename delayed_vector<T>::vector_type& delayed_vector<T>::data() const
{
	load();
	return m_data;
}


template<typename T>
void delayed_vector<T>::load() const
{
	if(m_delayData)
	{
		switch(m_delayData->m_fileType)
		{
		case FILE_BOOST_BINARY:
		{
			std::ifstream fs(m_delayData->m_filepath.c_str());
			boost::archive::binary_iarchive ar(fs);
			fs.seekg(m_delayData->m_fpos);

			assert(m_data.size() == 0);
			m_data.resize(m_delayData->m_size);

			ar >> boost::serialization::make_array(&m_data[0], m_data.size());
		}
		break;
		default:
		{
			// should not get here
			assert(false);
		}
		break;
		}

		m_delayData.reset();
	}
}


template<typename T>
std::size_t delayed_vector<T>::size() const
{
	return (m_delayData)? m_delayData->m_size : m_data.size();
}


template<typename T>
std::size_t delayed_vector<T>::clientSize() const
{
	return m_data.size();
}


template<typename T>
void delayed_vector<T>::resize(std::size_t size, bool destructive)
{
	if(destructive || (size == 0))
		m_delayData.reset();
	else
		load();

	m_data.resize(size);
}


template<typename T>
void delayed_vector<T>::shrink()
{
	if(m_delayData)
		return;

	if(m_data.capacity() > m_data.size())
		vector_type(m_data).swap(m_data);
}


template<typename T>
template<class Archive>
void delayed_vector<T>::save(Archive& ar, const unsigned int version) const
{
	namespace bs = boost::serialization;

	load();
	save_(ar, version, typename bs::use_array_optimization<Archive>::template apply<T>::type());
}


template<typename T>
template<class Archive>
void delayed_vector<T>::save_(Archive& ar, const unsigned int version, boost::mpl::false_) const
{
	using boost::serialization::make_nvp;
	ar & make_nvp("data", m_data);
}


template<typename T>
template<class Archive>
void delayed_vector<T>::save_(Archive& ar, const unsigned int version, boost::mpl::true_) const
{
	namespace bs = boost::serialization;

    const bs::collection_size_type count(m_data.size());
    ar << BOOST_SERIALIZATION_NVP(count);
    if (!m_data.empty())
    	ar << bs::make_array(&m_data[0], count);
}


template<typename T>
template<class Archive>
void delayed_vector<T>::load(Archive& ar, const unsigned int version)
{
	namespace bs = boost::serialization;

	m_delayData.reset();
	load_(ar, version, typename bs::use_array_optimization<Archive>::template apply<T>::type());
}


template<typename T>
template<class Archive>
void delayed_vector<T>::load_(Archive& ar, const unsigned int version, boost::mpl::false_)
{
	using boost::serialization::make_nvp;
	ar & make_nvp("data", m_data);
}


template<typename T>
template<class Archive>
void delayed_vector<T>::load_(Archive& ar, const unsigned int version, boost::mpl::true_)
{
	namespace bs = boost::serialization;

	bs::collection_size_type count;
	ar >> BOOST_SERIALIZATION_NVP(count);
	if(count == 0)
	{
		m_data.clear();
		return;
	}

	const detail::archive_info& ai = detail::archive_tracker::getInfo(ar);
	if(ai.m_loadOp.m_delayLoad)
	{
		m_data.clear();

		detail::DelayLoadData* dd = new detail::DelayLoadData();
		dd->m_size = count;
		loadDelayData(ar, ai, version, *dd, count);
		m_delayData.reset(dd);
	}
	else
	{
		m_data.resize(count);
		ar >> bs::make_array(&m_data[0], count);
	}
}


template<typename T>
void delayed_vector<T>::loadDelayData(boost::archive::binary_iarchive& ar, const detail::archive_info& ai,
	const unsigned int version, detail::DelayLoadData& dd, unsigned int numElems)
{
	assert(ai.m_fs);

	dd.m_fileType = FILE_BOOST_BINARY;
	dd.m_filepath = ai.m_filepath;
	dd.m_fpos = ai.m_fs->tellg();

	const unsigned int stride = sizeof(T) * numElems;

#ifdef NDEBUG
	// skip the data read. NOTE!!! This is sensitive to boost's implementation of array
	// serialisation, and in theory could change in future.
	ai.m_fs->seekg(stride, std::ios_base::cur);
#else
	// read the data into a temp vector
	vector_type tmpData(numElems);
	ar >> boost::serialization::make_array(&tmpData[0], numElems);

	std::streampos fpos2 = ai.m_fs->tellg();
	assert((fpos2 - dd.m_fpos) == stride);
#endif
}


}

#endif










