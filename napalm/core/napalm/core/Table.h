#ifndef _NAPALM_TABLE__H_
#define _NAPALM_TABLE__H_

#include "Buffer.h"
#include "TypedAttribute.h"
#include "exceptions.h"
#include "util/is_variant.hpp"
#include "meta.hpp"
//#include "List.h"
#include "impl/ValueWrapper.hpp"
#include <string>
#include <sstream>
#include <map>
#include <iterator>
#include <boost/variant.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <pystring.h>


namespace napalm {

	namespace detail {

		template<typename T>
		struct _is_keypath_1 {
			typedef typename boost::mpl::or_<boost::is_convertible<typename T::value_type,TableKey>,
					boost::is_convertible<typename T::value_type, const char *> >::type type;
		};

	} // detail ns


	/*
	 * If T is a container (such as std::vector) that can be used to describe a path into
	 * a napalm table then inherits from true_, otherwise inherits from false_.
	 */
	template<typename T>
	struct is_keypath : public boost::mpl::and_<
		is_iterable_sequence<T>,
		boost::mpl::not_<boost::is_same<T,std::string> >,
		detail::_is_keypath_1<T>
	>{};

	class keyCompare
	{
	public:
		bool operator()(const boost::shared_ptr<Attribute>& a, const boost::shared_ptr<Attribute>& b) const
		{
			return Dispatcher::instance().attribLessThan(*a, *b);
		}
	};

	/*
	 * @class Table
	 * @brief
	 * A table of values, keyable by attributes.
	 */
	template<typename Value>
	class Table : public Object
	{
		BOOST_MPL_ASSERT((boost::is_base_of<Object,Value>));

	public:

		typedef Value										value_class_type;
		typedef boost::shared_ptr<Value>					value_ptr;
		typedef boost::shared_ptr<Attribute>				key_ptr;

		typedef std::map<key_ptr, value_ptr, keyCompare> 	map_type;
		typedef typename map_type::value_type				value_type;
	    typedef typename map_type::iterator					map_iterator;
	    typedef typename map_type::const_iterator			map_const_iterator;

	    // const_iterator
		class const_iterator
		{
		public:
		    typedef std::pair<const key_ptr, value_ptr>		value_type;
		    typedef int										difference_type;
		    typedef value_type*   							pointer;
		    typedef value_type& 							reference;
		    typedef std::bidirectional_iterator_tag 		iterator_category;
		    friend class Table;



		protected:
			const_iterator( map_iterator a_Iter )
				: m_Iter ( a_Iter )
			{

			}

		public:
			const_iterator()
			{
			}

			const_iterator( const const_iterator& a_CI )
				: m_Iter( a_CI.m_Iter )
			{

			}

			virtual ~const_iterator() { }

			const_iterator & operator--()
			{
				--m_Iter;
				return *this;
			}

			const_iterator & operator++()
			{
				++m_Iter;
				return *this;
			}

			const_iterator operator++(int)
			{
				const_iterator old( m_Iter );
				++m_Iter;
				return old;
			}

			const std::pair<const key_ptr, value_ptr>& operator*()
			{
				return *m_Iter;
			}

			const std::pair<const key_ptr, value_ptr>* operator->()
			{
				return &(*m_Iter);
			}

			bool operator!=( const const_iterator & a_Other )
			{
				return m_Iter != a_Other.m_Iter;
			}

			bool operator==( const const_iterator & a_Other )
			{
				return m_Iter == a_Other.m_Iter;
			}

			template<typename T>
			bool getValue(T& value) const
			{
				return _getEntry( m_Iter, value );
			}

			template<typename T>
			bool extractValue( T& value )
			{
				boost::shared_ptr<Attribute> attrib =
					boost::dynamic_pointer_cast<Attribute>(m_Iter->second);
				if( !attrib )
				{
					return false;
				}

				return Dispatcher::instance().extractAttribValue( *attrib, value );
			}

			template<typename T>
			typename boost::enable_if<is_napalm_attrib_type<T>, T*>::type
			getValue()
			{
				boost::shared_ptr<TypedAttribute<T> > attrib =
					boost::dynamic_pointer_cast<TypedAttribute<T> >(m_Iter->second);
				return attrib ? &(attrib->value()) : NULL;
			}

			template<typename T>
			typename boost::disable_if<is_napalm_attrib_type<T>, boost::shared_ptr<T> >::type
			getValue()
			{
				return m_Iter->second;
			}

			template<typename T>
			typename boost::enable_if<is_napalm_attrib_type<T>, bool>::type
			getKey(T& value) const
			{
				boost::shared_ptr<TypedAttribute<T> > attrib =
					boost::dynamic_pointer_cast<TypedAttribute<T> >(m_Iter->first.get());
				if( attrib )
				{
					value = attrib->value();
					return true;
				}
				return false;
			}

			template<typename T>
			bool extractKey( T& value )
			{
				boost::shared_ptr<Attribute> attrib =
					boost::dynamic_pointer_cast<Attribute>(m_Iter->first.get());
				if( !attrib )
				{
					return false;
				}

				return Dispatcher::instance().extractAttribValue( *attrib, value );
			}

			template<typename T>
			typename boost::enable_if<is_napalm_attrib_type<T>, T*>::type
			getKey()
			{
				boost::shared_ptr<TypedAttribute<T> > attrib =
					boost::dynamic_pointer_cast<TypedAttribute<T> >(m_Iter->first.get());
				return attrib ? &(attrib->value()) : NULL;
			}

		protected:
			map_iterator m_Iter;
		};

		// iterator
		class iterator : public const_iterator
		{
		    friend class Table;

		protected:

			iterator( map_iterator a_Iter )
				: const_iterator( a_Iter )
			{

			}
		public:
			iterator()
			{

			}

			iterator( const iterator& a_I )
				: const_iterator( a_I )
			{

			}

			// These need to be present in each subclass so that they don't return
			// the superclass type
			iterator & operator--()
			{
				--this->m_Iter;
				return *this;
			}

			iterator & operator++()
			{
				++this->m_Iter;
				return *this;
			}

			iterator operator++(int)
			{
				iterator old( this->m_Iter );
				++this->m_Iter;
				return old;
			}

			// Non-const specific functions
			template<typename T>
			void setValue( const T& a_Entry )
			{
				this->m_Iter->second.reset( new TypedAttribute<T>(a_Entry) );
			}

			std::pair<const key_ptr, value_ptr>& operator*()
			{
				return *this->m_Iter;
			}

			std::pair<const key_ptr, value_ptr>* operator->()
			{
				return &(*this->m_Iter);
			}
		};

		template<typename Key>
		class typed_const_iterator : public const_iterator
		{
			friend class Table;

		protected:

			typed_const_iterator( map_iterator a_Iter, Table<Value> & a_Table )
				: const_iterator( a_Iter )
				, m_Table( a_Table )
			{
				// Seek to the first match
				while( this->m_Iter != m_Table.end().m_Iter &&
					   typeid(Key) != this->m_Iter->first->type() )
				{
					++this->m_Iter;
				}
			}
		public:

			// These must exist for each subclass to return the correct type
			// specially overridden for this type
			typed_const_iterator & operator--()
			{
				do {
				--this->m_Iter;
				} while( this->m_Iter != m_Table.begin().m_Iter &&
						typeid(Key) != this->m_Iter->first->type() );
				return *this;
			}

			typed_const_iterator & operator++()
			{
				do {
				++this->m_Iter;
				} while( this->m_Iter != m_Table.end().m_Iter &&
						typeid(Key) != this->m_Iter->first->type() );
				return *this;
			}

			typed_const_iterator operator++(int)
			{
				typed_const_iterator old( this->m_Iter, m_Table );
				do {
				++this->m_Iter;
				} while( this->m_Iter != m_Table.end().m_Iter &&
						typeid(Key) != this->m_Iter->first->type() );
				return old;
			}

			const Key & key() const
			{
				return (static_cast<TypedAttribute<Key> *>(this->m_Iter->first.get()))->value();
			}

		protected:
			Table<Value> & m_Table;
		};

		template<typename Key>
		class typed_iterator : public typed_const_iterator<Key>
		{
			friend class Table;

		protected:

			typed_iterator( map_iterator a_Iter, Table<Value> & a_Table )
				: typed_const_iterator<Key>( a_Iter, a_Table )
			{
			}
		public:

			// These must exist for each subclass to return the correct type
			// specially overridden for this type
			typed_iterator & operator--()
			{
				typed_const_iterator<Key>::operator--();
				return *this;
			}

			typed_iterator & operator++()
			{
				typed_const_iterator<Key>::operator++();
				return *this;
			}

			typed_iterator operator++(int i)
			{
				typed_iterator old( this->m_Iter, this->m_Table, this->m_Type );
				typed_const_iterator<Key>::operator++();
				return old;
			}


			// Non-const specific functions
			template<typename T>
			void setValue( const T& a_Entry )
			{
				this->m_Iter->second.reset( new TypedAttribute<T>(a_Entry) );
			}

			std::pair<const key_ptr, value_ptr>& operator*()
			{
				return *this->m_Iter;
			}

			std::pair<const key_ptr, value_ptr>* operator->()
			{
				return &(*this->m_Iter);
			}
		};


		Table(){}
		virtual ~Table(){}


		/*
		 * std::map-like interface
		 */

		iterator begin() 										{ return iterator( m_map.begin() ); }

		const_iterator begin() const 							{ return const_iterator( const_cast<map_type&>(m_map).begin() ); }

		iterator end()											{ return iterator( m_map.end() ); }

		const_iterator end() const								{ return const_iterator( const_cast<map_type&>(m_map).end() ); }

		template<typename Key>
		typed_iterator<Key> typed_begin() 						{ return typed_iterator<Key>( m_map.begin(), *this ); }

		template<typename Key>
		typed_const_iterator<Key> typed_begin() const 			{ return typed_const_iterator<Key>( const_cast<map_type&>(m_map).begin(),
																							   *const_cast<Table*>(this) ); }

		value_ptr& operator[](const TableKey& key)				{ return m_map[key.get()]; }

		bool empty() const										{ return m_map.empty(); }
		unsigned int size() const								{ return m_map.size(); }

		std::pair<iterator,bool> insert(const value_type& x)
		{
			std::pair<map_iterator, bool> p = m_map.insert(x);
			return std::pair<iterator,bool>(iterator(p.first), p.second);
		}

		void erase(iterator pos)								{ m_map.erase(pos.m_Iter); }
		bool erase(const TableKey& key)							{ return (m_map.erase(key.get()) > 0); }

		void clear()											{ m_map.clear(); }

		iterator find(const TableKey& key)						{ return iterator( m_map.find(key.get()) ); }

		const_iterator find(const TableKey& key) const			{ return const_iterator( const_cast<map_type&>(m_map).find(key.get()) ); }

		/*
		 * @brief hasEntry
		 * Query whether a table entry exists.
		 * @param key The search key.
		 * @returns True if the key was found, false otherwise.
		 */
		bool hasEntry(const TableKey& key) const;


		template<typename KeyContainer>
		typename boost::enable_if<is_keypath<KeyContainer>, bool>::type
		/*bool*/ hasEntryPath(const KeyContainer& keypath) const
		{
			return hasEntryPath(keypath.begin(), keypath.end());
		}

		template<typename KeyIterator>
		bool hasEntryPath(KeyIterator keypathBegin, KeyIterator keypathEnd) const;

		bool hasEntryPathT( const TableKey & p0 = TableKey(), const TableKey & p1 = TableKey(),
						   const TableKey & p2 = TableKey(), const TableKey & p3 = TableKey(), const TableKey & p4 = TableKey(),
						   const TableKey & p5 = TableKey(), const TableKey & p6 = TableKey(), const TableKey & p7 = TableKey() )
		{
			std::vector<TableKey> keys;
			if( !appendIfPresent( keys, p0 ) ) { return hasEntryPath(keys); }
			if( !appendIfPresent( keys, p1 ) ) { return hasEntryPath(keys); }
			if( !appendIfPresent( keys, p2 ) ) { return hasEntryPath(keys); }
			if( !appendIfPresent( keys, p3 ) ) { return hasEntryPath(keys); }
			if( !appendIfPresent( keys, p4 ) ) { return hasEntryPath(keys); }
			if( !appendIfPresent( keys, p5 ) ) { return hasEntryPath(keys); }
			if( !appendIfPresent( keys, p6 ) ) { return hasEntryPath(keys); }
			appendIfPresent( keys, p7 );
			return hasEntryPath( keys );
		}

		static bool appendIfPresent( std::vector<TableKey> & keys, const TableKey & arg )
		{
			if( !arg.isNull() )
			{
				keys.push_back( arg );
				return true;
			}
			return false;
		}

		template< typename T >
		bool getEntryPathT( T & a_Out, const TableKey & p0 = TableKey(), const TableKey & p1 = TableKey(),
		                   const TableKey & p2 = TableKey(), const TableKey & p3 = TableKey(), const TableKey & p4 = TableKey(),
						   const TableKey & p5 = TableKey(), const TableKey & p6 = TableKey(), const TableKey & p7 = TableKey() )
		{
			std::vector<TableKey> keys;
			if( !appendIfPresent( keys, p0 ) ) { return getEntryPath(keys, a_Out); }
			if( !appendIfPresent( keys, p1 ) ) { return getEntryPath(keys, a_Out); }
			if( !appendIfPresent( keys, p2 ) ) { return getEntryPath(keys, a_Out); }
			if( !appendIfPresent( keys, p3 ) ) { return getEntryPath(keys, a_Out); }
			if( !appendIfPresent( keys, p4 ) ) { return getEntryPath(keys, a_Out); }
			if( !appendIfPresent( keys, p5 ) ) { return getEntryPath(keys, a_Out); }
			if( !appendIfPresent( keys, p6 ) ) { return getEntryPath(keys, a_Out); }
			appendIfPresent( keys, p7 );
			return getEntryPath( keys, a_Out );
		}

		/*
		 * @brief delEntry
		 * Delete a value from the table.
		 * @tparam T T can be either of:
		 * - a base attribute type (such as int, float, v3f etc);
		 * - boost::shared_ptr<[const] S>, where S is any Object-derived class;
		 * - boost::variant<...>, where the variant types are any of the above.
		 * @param key The search key.
		 * @returns True if the entry was found and deleted, false otherwise.
		 */
		bool delEntry(const TableKey& key);

		template<typename KeyContainer>
		typename boost::enable_if<is_keypath<KeyContainer>, bool>::type
		/*bool*/ delEntryPath(const KeyContainer& keypath)
		{
			return delEntryPath(keypath.begin(), keypath.end());
		}

		template<typename KeyIterator>
		bool delEntryPath(KeyIterator keypathBegin, KeyIterator keypathEnd);

		/*
		 * @brief getEntry
		 * Get a value from the table.
		 * @tparam T T can be either of:
		 * - a base attribute type (such as int, float, v3f etc);
		 * - boost::shared_ptr<[const] S>, where S is any Object-derived class;
		 * - boost::variant<...>, where the variant types are any of the above.
		 * @param key The search key.
		 * @param value The entry value (unchanged if entry is not found).
		 * @returns True if the entry is found and there is a type match, false otherwise.
		 */
		template<typename T>
		bool getEntry(const TableKey& key, T& value) const;

		template<typename KeyContainer, typename T>
		typename boost::enable_if<is_keypath<KeyContainer>, bool>::type
		/*bool*/ getEntryPath(const KeyContainer& keypath, T& value) const
		{
			return getEntryPath(keypath.begin(), keypath.end(), value);
		}

		template<typename KeyIterator, typename T>
		bool getEntryPath(KeyIterator keypathBegin, KeyIterator keypathEnd, T& value) const;

		/*
		 * @brief setEntry
		 * Set a value in the table, overwriting the entry if it already exists.
		 * @tparam T T can be either of:
		 * - a base attribute type (such as int, float, v3f etc);
		 * - boost::shared_ptr<S>, where S is any Value-derived class;
		 * - boost::variant<...>, where the variant types are any of the above.
		 * @param key The entry to write to.
		 * @param value The value to write.
		 */
		template<typename T>
		void setEntry( const TableKey& key, const T& value);

		template<typename KeyContainer, typename T>
		typename boost::enable_if<is_keypath<KeyContainer>, bool>::type
		/*bool*/ setEntryPath(const KeyContainer& keypath, const T& value, bool createPath = false)
		{
			return setEntryPath(keypath.begin(), keypath.end(), value, createPath);
		}

		template<typename KeyIterator, typename T>
		bool setEntryPath(KeyIterator keypathBegin, KeyIterator keypathEnd,
			const T& value, bool createPath = false);

		template<typename T>
		typename boost::enable_if<is_napalm_attrib_type<T>, T*>::type
		getEntry( const TableKey & key );

		template<typename T>
		typename boost::disable_if<is_napalm_attrib_type<T>, boost::shared_ptr<T> >::type
		getEntry( const TableKey & key );

		template< typename T >
		bool setEntryPathT( T & a_Out, const TableKey & p0 = TableKey(), const TableKey & p1 = TableKey(),
						   const TableKey & p2 = TableKey(), const TableKey & p3 = TableKey(), const TableKey & p4 = TableKey(),
						   const TableKey & p5 = TableKey(), const TableKey & p6 = TableKey(), const TableKey & p7 = TableKey() )
		{
			std::vector<TableKey> keys;
			if( !appendIfPresent( keys, p0 ) ) { return setEntryPath(keys, a_Out); }
			if( !appendIfPresent( keys, p1 ) ) { return setEntryPath(keys, a_Out); }
			if( !appendIfPresent( keys, p2 ) ) { return setEntryPath(keys, a_Out); }
			if( !appendIfPresent( keys, p3 ) ) { return setEntryPath(keys, a_Out); }
			if( !appendIfPresent( keys, p4 ) ) { return setEntryPath(keys, a_Out); }
			if( !appendIfPresent( keys, p5 ) ) { return setEntryPath(keys, a_Out); }
			if( !appendIfPresent( keys, p6 ) ) { return setEntryPath(keys, a_Out); }
			appendIfPresent( keys, p7 );
			return setEntryPath( keys, a_Out );
		}

		template<typename T>
		bool extractEntry( const TableKey & key, T& value );

		template<typename T>
		T extractEntry( const TableKey & key );

		template<typename T>
		T extractEntryWithDefault( const TableKey & key, const T & defaultValue );

		virtual object_ptr clone(object_clone_map& cloned) const;

		virtual std::ostream& str(std::ostream& os, object_rawptr_set& printed, util::StringMode a_Type = util::DEFAULT) const;

		virtual std::ostream& dump(std::ostream& os, object_rawptr_set& printed) const;

		// just like dump, but with an option not to print the leading type string
		std::ostream& dump2(std::ostream& os, object_rawptr_set& printed, bool printType) const;

	protected:

		template<typename Variant>
		struct get_entry_dispatcher
		{
			struct info
			{
				info(map_const_iterator it, Variant& value)
				: m_it(it), m_value(value), m_match(false){}

				map_const_iterator m_it;
				Variant& m_value;
				bool m_match;
			};

			get_entry_dispatcher(info& i)
			: m_info(i)
			{
				boost::mpl::for_each<typename Variant::types>(*this);
			}

			template<typename T>
			void operator()(T x)
			{
				if(!m_info.m_match)
				{
					T v;
					if(_getEntryNonVariant(m_info.m_it, v))
					{
						m_info.m_value = v;
						m_info.m_match = true;
					}
				}
			}

			info& m_info;
		};

		struct set_entry_visitor : public boost::static_visitor<void>
		{
			set_entry_visitor(Table& table, const TableKey& key)
			: m_table(table), m_key(key){}

			template<typename T>
			void operator()(const T& value) const
			{
				m_table._setEntryNonVariant(m_key, value);
			}

			Table& m_table;
			const TableKey& m_key;
		};

		template<typename Variant>
		friend class get_entry_dispatcher;

		friend class set_entry_visitor;
		friend class iterator;
		friend class const_iterator;

		// getEntry delegation
		template<typename T>
		static typename boost::disable_if<util::is_variant<T>, bool>::type
		/*bool*/ _getEntry(map_const_iterator it, T& value);

		template<typename T>
		static typename boost::enable_if<util::is_variant<T>, bool>::type
		/*bool*/ _getEntry(map_const_iterator it, T& value);

		template<typename T>
		static bool _getEntryNonVariant(map_const_iterator it, T& value);

		template<typename T>
		static bool _getEntryNonVariant(map_const_iterator it, boost::shared_ptr<T>& value);

		// setEntry delegation
		template<typename T>
		void _setEntry(const TableKey& key, const T& value);

		void _setEntry(const TableKey& key, const char* value);

		template<typename T>
		typename boost::disable_if<util::is_variant<T>, void>::type
		__setEntry(const TableKey& key, const T& value);

		template<typename T>
		typename boost::enable_if<util::is_variant<T>, void>::type
		__setEntry(const TableKey& key, const T& value);

		template<typename S, typename Q>
		typename boost::disable_if< is_napalm_attrib_type< std::vector<S, Q> >, void >::type
		_setEntryNonVariant(const TableKey& key, const std::vector<S, Q>& value);

		template<typename S, typename Q, typename R>
		typename boost::disable_if< is_napalm_attrib_type< std::set<S, Q, R> >, void >::type
		_setEntryNonVariant(const TableKey& key, const std::set<S, Q, R>& value);

		template<typename T>
		typename boost::enable_if< is_napalm_attrib_type< T >, void >::type
		_setEntryNonVariant(const TableKey& key, const T& value);

		template<typename T>
		void _setEntryNonVariant(const TableKey& key, const boost::shared_ptr<T>& value);


		// boost serialization
		friend class boost::serialization::access;
		template<class Archive> void serialize(Archive& ar, const unsigned int version);

	protected:

		map_type m_map;
	};


///////////////////////// impl

namespace detail {

	struct set_table_attrib_entry_visitor : public boost::static_visitor<bool>
	{
		set_table_attrib_entry_visitor(AttribTable& atable, const TableKey& key)
		: m_atable(atable), m_key(key){}

		template<typename T>
		bool operator()(const T& value) const {
			return setAttribEntry(m_atable, m_key, value);
		}

		AttribTable& m_atable;
		const TableKey& m_key;
	};

	template<typename T>
	typename boost::disable_if<is_napalm_base_type<T>, bool>::type
	setAttribEntry(AttribTable& atable, const TableKey& key, const T& value)
	{
		return false;
	}

	template<typename T>
	typename boost::enable_if<is_napalm_base_type<T>, bool>::type
	setAttribEntry(AttribTable& atable, const TableKey& key, const T& value)
	{
		atable.setEntry(key, value);
		return true;
	}

	template<typename T>
	bool setAttribEntry(AttribTable& atable, const TableKey& key,
		const boost::shared_ptr<TypedAttribute<T> >& value)
	{
		atable.setEntry(key, value);
		return true;
	}

	template<BOOST_VARIANT_ENUM_PARAMS(typename T)>
	bool setAttribEntry(AttribTable& atable, const TableKey& key,
		const boost::variant<BOOST_VARIANT_ENUM_PARAMS(T)>& value)
	{
		set_table_attrib_entry_visitor vis(atable, key);
		return boost::apply_visitor(vis, value);
	}

	bool setAttribEntry(AttribTable& atable, const TableKey& key, const attrib_ptr value);

} // detail ns


/////////////////////////
// hasEntry
/////////////////////////

template<typename Value>
bool Table<Value>::hasEntry(const TableKey& key) const
{
	return (m_map.find(key.get()) != m_map.end());
}


template<typename Value>
template<typename KeyIterator>
bool Table<Value>::hasEntryPath(KeyIterator keypathBegin, KeyIterator keypathEnd) const
{
	c_object_ptr obj;
	return getEntryPath(keypathBegin, keypathEnd, obj);
}


/////////////////////////
// getEntry
/////////////////////////

template<typename Value>
template<typename T>
bool Table<Value>::getEntry(const TableKey& key, T& value) const
{
	map_const_iterator it = m_map.find(key.get());
	if(it == m_map.end())
		return false;

	return _getEntry(it, value);
}


// recursive getEntry
template<typename Value>
template<typename KeyIterator, typename T>
bool Table<Value>::getEntryPath(KeyIterator keypathBegin, KeyIterator keypathEnd, T& value) const
{
	if(keypathBegin == keypathEnd)
		throw NapalmError("empty keypath");

	KeyIterator begin2 = keypathBegin;
	++begin2;

	if(begin2 == keypathEnd)
		return getEntry(*keypathBegin, value);

	c_object_ptr obj;
	if(getEntry(*keypathBegin, obj))
	{
		c_object_table_ptr table = boost::dynamic_pointer_cast<const ObjectTable>(obj);
		if(table)
			return table->getEntryPath(begin2, keypathEnd, value);

		c_buffer_ptr buf = boost::dynamic_pointer_cast<const Buffer>(obj);
		c_attrib_table_ptr atable = (buf)?
			buf->getAttribs() : boost::dynamic_pointer_cast<const AttribTable>(obj);

		if(atable)
			return atable->getEntryPath(begin2, keypathEnd, value);
	}

	return false;
}

template<typename Value>
template<typename T>
typename boost::enable_if<is_napalm_attrib_type<T>, T*>::type
Table<Value>::getEntry( const TableKey & key )
{
	map_const_iterator it = m_map.find(key.get());
	if(it == m_map.end())
		return NULL;
	boost::shared_ptr<TypedAttribute<T> > attrib =
		boost::dynamic_pointer_cast<TypedAttribute<T> >(it->second);
	return attrib ? &(attrib->value()) : NULL;
}

template<typename Value>
template<typename T>
typename boost::disable_if<is_napalm_attrib_type<T>, boost::shared_ptr<T> >::type
Table<Value>::getEntry( const TableKey & key )
{
	boost::shared_ptr<T> ptr;
	getEntry( key, ptr );
	return ptr;
}

template<typename Value>
template<typename T>
bool Table<Value>::extractEntry( const TableKey & key, T& value )
{
	map_const_iterator it = m_map.find(key.get());
	if( it == m_map.end() )
	{
		return false;
	}
	boost::shared_ptr<Attribute> attrib =
		boost::dynamic_pointer_cast<Attribute>(it->second);
	if( !attrib )
	{
		return false;
	}

	return Dispatcher::instance().extractAttribValue( *attrib, value );
}

template<typename Value>
template<typename T>
T Table<Value>::extractEntry( const TableKey & key )
{
	T rv;
	if( !extractEntry(key, rv) )
	{
		throw NapalmError("Can't extract entry");
	}
	return rv;
}

template<typename Value>
template<typename T>
T Table<Value>::extractEntryWithDefault( const TableKey & key, const T & defaultValue )
{
	T rv;
	if( !extractEntry(key, rv) )
	{
		return defaultValue;
	}
	return rv;
}

// getEntry attrib base type specialization
template<typename Value>
template<typename T>
/*static*/ bool Table<Value>::_getEntryNonVariant(map_const_iterator it, T& value)
{
	boost::shared_ptr<TypedAttribute<T> > attrib =
		boost::dynamic_pointer_cast<TypedAttribute<T> >(it->second);

	if(attrib)
	{
		value = attrib->value();
		return true;
	}

	return false;
}


// getEntry boost::shared_ptr<T> specialization
template<typename Value>
template<typename T>
/*static*/ bool Table<Value>::_getEntryNonVariant(map_const_iterator it,
	boost::shared_ptr<T>& value)
{
	typedef typename boost::remove_cv<T>::type T_;

	boost::shared_ptr<T_> obj =
		boost::dynamic_pointer_cast<T_>(it->second);

	if(obj)
	{
		value = obj;
		return true;
	}

	return false;
}


// getEntry NOT(boost::variant) specialization
template<typename Value>
template<typename T>
typename boost::disable_if<util::is_variant<T>, bool>::type
/*static bool*/ Table<Value>::_getEntry(map_const_iterator it, T& value)
{
	return _getEntryNonVariant(it, value);
}


// getEntry boost::variant specialization
template<typename Value>
template<typename Variant>
typename boost::enable_if<util::is_variant<Variant>, bool>::type
/*static bool*/ Table<Value>::_getEntry(map_const_iterator it, Variant& value)
{
	typename get_entry_dispatcher<Variant>::info i(it, value);
	get_entry_dispatcher<Variant> vis(i);
	return i.m_match;
}


/////////////////////////
// delEntry
/////////////////////////

template<typename Value>
bool Table<Value>::delEntry(const TableKey& key)
{
	map_iterator it = m_map.find(key.get());
	if(it == m_map.end())
		return false;

	m_map.erase(it);
	return true;
}


// recursive delEntry
template<typename Value>
template<typename KeyIterator>
bool Table<Value>::delEntryPath(KeyIterator keypathBegin, KeyIterator keypathEnd)
{
	if(keypathBegin == keypathEnd)
		throw NapalmError("empty keypath");

	KeyIterator begin2 = keypathBegin;
	++begin2;

	if(begin2 == keypathEnd)
		return delEntry(*keypathBegin);

	object_ptr obj;
	if(getEntry(*keypathBegin, obj))
	{
		object_table_ptr table = boost::dynamic_pointer_cast<ObjectTable>(obj);
		if(table)
			return table->delEntryPath(begin2, keypathEnd);

		buffer_ptr buf = boost::dynamic_pointer_cast<Buffer>(obj);
		attrib_table_ptr atable = (buf)?
			buf->getAttribs() : boost::dynamic_pointer_cast<AttribTable>(obj);

		if(atable)
			return atable->delEntryPath(begin2, keypathEnd);
	}

	return false;
}


/////////////////////////
// setEntry
/////////////////////////

template<typename Value>
template<typename T>
void Table<Value>::setEntry(const TableKey& key, const T& value)
{
	_setEntry(key, value);
}


// recursive setEntry (ObjectTable specialization)
template<>
template<typename KeyIterator, typename T>
bool Table<Object>::setEntryPath(KeyIterator keypathBegin, KeyIterator keypathEnd,
	const T& value, bool createPath)
{
	if(keypathBegin == keypathEnd)
		throw NapalmError("empty keypath");

	KeyIterator begin2 = keypathBegin;
	++begin2;

	if(begin2 == keypathEnd)
	{
		setEntry(*keypathBegin, value);
		return true;
	}

	object_ptr obj;
	if(getEntry(*keypathBegin, obj))
	{
		object_table_ptr table = boost::dynamic_pointer_cast<ObjectTable>(obj);
		if(table)
			return table->setEntryPath(begin2, keypathEnd, value, createPath);

		buffer_ptr buf = boost::dynamic_pointer_cast<Buffer>(obj);
		attrib_table_ptr atable = (buf)?
			buf->getAttribs() : boost::dynamic_pointer_cast<AttribTable>(obj);

		if(atable && (std::distance(begin2, keypathEnd) == 1))
		{
			if(detail::setAttribEntry(*atable.get(), *begin2, value))
				return true;
		}
	}

	if(createPath)
	{
		object_table_ptr newTable(new ObjectTable());
		setEntry(*keypathBegin, newTable);
		return newTable->setEntryPath(begin2, keypathEnd, value, createPath);
	}

	return false;
}


// recursive setEntry (AttribTable specialization)
template<>
template<typename KeyIterator, typename T>
bool Table<Attribute>::setEntryPath(KeyIterator keypathBegin, KeyIterator keypathEnd,
	const T& value, bool createPath)
{
	if(std::distance(keypathBegin, keypathEnd) > 1)
		return false;

	setEntry(*keypathBegin, value);
	return true;
}



template<typename Value>
template<typename S, typename Q>
typename boost::disable_if< is_napalm_attrib_type< std::vector<S, Q> >, void >::type
Table<Value>::_setEntryNonVariant(const TableKey& key, const std::vector<S, Q>& value)
{
	m_map[key.get()] = attrib_ptr(new TypedAttribute<std::vector<S, util::counted_allocator<S> > >(value));
}

template<typename Value>
template<typename S, typename Q, typename R>
typename boost::disable_if< is_napalm_attrib_type< std::set<S, Q, R> >, void >::type
Table<Value>::_setEntryNonVariant(const TableKey& key, const std::set<S, Q, R>& value)
{
	m_map[key.get()] = attrib_ptr(new TypedAttribute<std::set<S, util::less<S>, util::counted_allocator<S> > >(value));
}

// setEntry attrib base type specialization
template<typename Value>
template<typename T>
typename boost::enable_if< is_napalm_attrib_type< T >, void >::type
Table<Value>::_setEntryNonVariant(const TableKey& key, const T& value)
{
	m_map[key.get()] = attrib_ptr(new TypedAttribute<T>(value));
}


// setEntry boost::shared_ptr<T> specialization
template<typename Value>
template<typename T>
void Table<Value>::_setEntryNonVariant(const TableKey& key, const boost::shared_ptr<T>& value)
{
	m_map[key.get()] = value;
}


template<typename Value>
template<typename T>
typename boost::disable_if<util::is_variant<T>, void>::type
Table<Value>::__setEntry(const TableKey& key, const T& value)
{
	_setEntryNonVariant(key, value);
}


template<typename Value>
template<typename Variant>
typename boost::enable_if<util::is_variant<Variant>, void>::type
Table<Value>::__setEntry(const TableKey& key, const Variant& value)
{
	set_entry_visitor vis(*this, key);
	boost::apply_visitor(vis, value);
}


// setEntry const char* specialization
template<typename Value>
void Table<Value>::_setEntry(const TableKey& key, const char* value)
{
	m_map[key.get()] = attrib_ptr(new TypedAttribute<std::string>(value));
}


template<typename Value>
template<typename T>
void Table<Value>::_setEntry(const TableKey& key, const T& value)
{
	__setEntry(key, value);
}


/////////////////////////
// other
/////////////////////////

template<typename Value>
object_ptr Table<Value>::clone(object_clone_map& cloned) const
{
	assert(cloned.find(this) == cloned.end());

	Table* pclone = new Table();
	for(const_iterator it = begin(); it!=end(); ++it)
		pclone->insert(typename map_type::value_type(it->first, make_clone(it->second, &cloned)));

	boost::shared_ptr<Table> obj(pclone);
	cloned.insert(object_clone_map::value_type(this, obj));
	return obj;
}


template<typename Value>
std::ostream& Table<Value>::str(std::ostream& os, object_rawptr_set& printed, util::StringMode a_Type) const
{
	if(printed.find(this) == printed.end())
		printed.insert(this);
	else
		return strPtr(os);

	os << '{';
	bool first = true;
	for(map_const_iterator it = m_map.begin(); it!=m_map.end(); ++it, first=false)
	{
		if(!first)
			os << ", ";
		os << *it->first << ": ";

		if(it->second)
			it->second->str(os, printed, a_Type);
		else
			os << Object::nullRepr();
	}

	os << '}';
	return os;
}


template<typename Value>
std::ostream& Table<Value>::dump(std::ostream& os, object_rawptr_set& printed) const
{
	return dump2(os, printed, true);
}


template<typename Value>
std::ostream& Table<Value>::dump2(std::ostream& os,
	object_rawptr_set& printed, bool printType) const
{
	const std::size_t num_tabulation_spaces = 2;

	if(printed.find(this) == printed.end())
		printed.insert(this);
	else
		return (strPtr(os) << "*\n");

	std::size_t tabulation = 0;
	for(map_const_iterator it = m_map.begin(); it!=m_map.end(); ++it)
	{
		std::ostringstream strm;
		strm << *it->first;
		tabulation = std::max(tabulation, strm.str().length());
	}

	tabulation += num_tabulation_spaces + 1;
	std::string tabstr("\n");
	for(unsigned int i=0; i<tabulation; ++i)
		tabstr += " ";

	if(printType)
	{
		strPtr(os);
		os << '\n';
	}

	for(map_const_iterator it = m_map.begin(); it!=m_map.end(); ++it)
	{
		{
			std::ostringstream strm;
			strm << *it->first << ':';
			unsigned int nspaces = tabulation - strm.str().length();
			for(unsigned int i=0; i<nspaces; ++i)
				strm << ' ';
			os << strm.str();
		}

		if(it->second)
		{
			std::ostringstream strm;
			it->second->dump(strm, printed);
			std::string str = strm.str();
			unsigned int nreturns = static_cast<unsigned int>(pystring::count(str, "\n"));
			if(nreturns > 1)
				str = pystring::replace(strm.str(), "\n", tabstr, nreturns-1);

			os << str;
		}
		else
			os << Object::nullRepr();
	}

	return os;
}


template<typename Value>
template<class Archive>
void Table<Value>::serialize(Archive& ar, const unsigned int version)
{
	using boost::serialization::make_nvp;
	using boost::serialization::base_object;

	ar & make_nvp("base_class_object", base_object<Object>(*this));
	ar & make_nvp("map", m_map);
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
