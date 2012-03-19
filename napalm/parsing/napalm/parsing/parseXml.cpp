//#define BOOST_SPIRIT_DEBUG

#include "parseXml.h"
#include "napalm/core/exceptions.h"
#include "napalm/core/TypedAttribute.h"
#include <boost/spirit.hpp>
#include <stack>
#include <sstream>
#include <pystring.h>

const char* napalm::g_xml_tag_name = "__name__";
const char* napalm::g_xml_tag_type = "__type__";

const char* napalm::g_xml_type_root = "__root__";
const char* napalm::g_xml_type_element = "__element__";
const char* napalm::g_xml_type_array = "__array__";
const char* napalm::g_xml_type_value = "__value__";
const char* napalm::g_xml_type_attrib = "__attrib__";

boost::mutex  napalm::g_parse_mtx;

namespace bs = boost::spirit;

namespace napalm
{

	void getXmlTags( const object_table_ptr t, std::string & o_name, std::string & o_type )
	{
		bool ret = false;
		ret = t->getEntry(g_xml_tag_name,o_name);
		assert( ret && "Could not find '__name__' in table" );
		ret = t->getEntry(g_xml_tag_type,o_type);
		assert( ret && "Could not find '__type__' in table" );
	}

	void setXmlTags( object_table_ptr t, const std::string & a_name, const std::string & a_type )
	{
		t->setEntry(g_xml_tag_type,a_type);
		t->setEntry(g_xml_tag_name,a_name);
	}

	// grammar for xml parsing
	struct xml_grammar : public bs::grammar<xml_grammar>
	{
		template<class ScannerT>
		struct definition
		{
			typedef bs::chset<char> chset_t;
			bs::rule<ScannerT>
				Document,
				Value,
				Prolog,
				XMLDecl,
				Element,
				Content,
				Attribute,
				AttribValue,
				Name,
				STag,ETag,
				S,
				Eq;

			definition(const xml_grammar& g)
			{
				Document = Prolog >> Element;
				Value = !S >> (
							bs::strict_real_p [g.m_a_float] |
							bs::int_p [g.m_a_int] |
							( *( bs::anychar_p - '<' ) ) [g.m_a_str]
							) >> !S ;

				S = +bs::space_p;
				Eq = !S >> '=' >> !S;

				Prolog = (!XMLDecl) [g.m_a_prolog];
				XMLDecl = bs::lexeme_d[ "<?xml" >> *(bs::anychar_p-"?>") >> "?>" ];

				Name = ( bs::alpha_p >> *( bs::alnum_p | chset_t( "_:" ) ) );
				STag = ( '<' >> Name [g.m_a_stag_name] >> *( !S >> Attribute ) >> '>' );
				ETag = ( "</" >> Name [g.m_a_etag_name] >> !S >> '>' );
				Element = !S >>
						  (
							STag >>
							Content >>
							ETag
						  )	>> !S;
				Content = !Value >> *( Element >> !Value );
				Attribute = ( Name [g.m_a_attrib_name] >> Eq >> AttribValue  );
				AttribValue = '"' >> ( *( bs::anychar_p - ( chset_t( "<&\"" ) ) ) ) [g.m_a_attrib_value] >> '"' |
							  '\'' >> ( *( bs::anychar_p - ( chset_t( "<&'" ) ) ) ) [g.m_a_attrib_value] >> '\'';

				BOOST_SPIRIT_DEBUG_RULE(Document);
				BOOST_SPIRIT_DEBUG_RULE(Value);
				BOOST_SPIRIT_DEBUG_RULE(Prolog);
				BOOST_SPIRIT_DEBUG_RULE(XMLDecl);
				BOOST_SPIRIT_DEBUG_RULE(Element);
				BOOST_SPIRIT_DEBUG_RULE(Content);
				BOOST_SPIRIT_DEBUG_RULE(Attribute);
				BOOST_SPIRIT_DEBUG_RULE(AttribValue);
				BOOST_SPIRIT_DEBUG_RULE(Name);
				BOOST_SPIRIT_DEBUG_RULE(STag);
				BOOST_SPIRIT_DEBUG_RULE(ETag);
				BOOST_SPIRIT_DEBUG_RULE(S);
				BOOST_SPIRIT_DEBUG_RULE(Eq);
			}

			bs::rule<ScannerT> const &start() { return Document; }
		};

		struct action {
			xml_grammar& m_g;
			action(xml_grammar& g):m_g(g){}
		};

		struct action_int : public action
		{
			action_int(xml_grammar& g):action(g){}
			void operator()(int val) const
			{
				object_table_ptr top(boost::dynamic_pointer_cast<ObjectTable>(m_g.m_dom.top()));
				attrib_ptr xvalue;
				if (!top->getEntry(g_xml_type_value, xvalue))
					top->setEntry(g_xml_type_value,val);
				else
				{
					std::ostringstream strm;
					strm << xvalue << ' ' << val;
					top->setEntry(g_xml_type_value,pystring::strip(strm.str()));
				}
			}
		};

		struct action_float : public action
		{
			action_float(xml_grammar& g):action(g){}
			void operator()(double val) const
			{
				object_table_ptr top(boost::dynamic_pointer_cast<ObjectTable>(m_g.m_dom.top()));
				attrib_ptr xvalue;
				if (!top->getEntry(g_xml_type_value, xvalue))
					top->setEntry(g_xml_type_value,val);
				else
				{
					std::ostringstream strm;
					strm << xvalue << ' ' << val;
					top->setEntry(g_xml_type_value,pystring::strip(strm.str()));
				}
			}
		};

		struct action_str : public action
		{
			action_str(xml_grammar& g):action(g){}
			void operator()(const char* begin, const char* end) const
			{
				if ( begin == end )
					return;
				std::string s(begin, end-begin);
				object_table_ptr top(boost::dynamic_pointer_cast<ObjectTable>(m_g.m_dom.top()));
				attrib_ptr xvalue;
				if (!top->getEntry(g_xml_type_value, xvalue))
					top->setEntry(g_xml_type_value,pystring::strip(s));
				else
				{
					std::ostringstream strm;
					strm << xvalue << ' ' << s;
					top->setEntry(g_xml_type_value,pystring::strip(strm.str()));
				}
			}
		};

		struct action_prolog : public action
		{
			action_prolog(xml_grammar& g):action(g){}
			void operator()(const char* begin, const char* end) const
			{
				assert(m_g.m_dom.empty());
				object_table_ptr t(new ObjectTable());
				setXmlTags(t, g_xml_type_root, g_xml_type_root);
				m_g.m_dom.push(t);
			}
		};

		struct action_stag_name : public action
		{
			action_stag_name(xml_grammar& g):action(g){}
			void operator()(const char* begin, const char* end) const
			{
				assert(!m_g.m_dom.empty());

				std::string elem_name(begin, end-begin);
				object_table_ptr t(new ObjectTable());
				setXmlTags(t, elem_name, g_xml_type_element);

				object_table_ptr top(boost::dynamic_pointer_cast<ObjectTable>(m_g.m_dom.top()));
				std::string top_name;
				std::string top_type_value;
				getXmlTags(top, top_name, top_type_value);

				// if our top is the same as name as us AND is an element, then
				// we need to create a new (array) table, add the current top to it,
				// replace the top with the new array-table and append 't'
				if (elem_name == top_name)
				{
					if (top_type_value == g_xml_type_array)
					{
						int key = 0;
						for( Table<Object>::const_iterator it = top->begin();
								it != top->end(); ++it )
						{
						    int *p = it.getKey<int>();
						    if( p && *p >= key )
						    {
					            key = *p + 1;
						    }
				        }

						top->setEntry( key, t);
					}
					else
					{
						assert( m_g.m_dom.size() > 1 );
						object_table_ptr array_table(new ObjectTable());
						setXmlTags(array_table, elem_name, g_xml_type_array);
						array_table->setEntry(0,top);
						array_table->setEntry(1,t);
						m_g.m_dom.pop();
						object_table_ptr topParent(boost::dynamic_pointer_cast<ObjectTable>(m_g.m_dom.top()));
						topParent->setEntry( elem_name, array_table);
						m_g.m_dom.push(array_table);
					}
				}
				else
				{
					object_table_ptr named;
					if ( top->getEntry(elem_name, named) )
					{
						std::string array_table_type_value;
						std::string array_table_name;
						getXmlTags(named, array_table_name, array_table_type_value);

						if (array_table_type_value == g_xml_type_array)
						{
							int key = 0;
							for( Table<Object>::const_iterator it = named->begin();
									it != named->end(); ++it )
							{
							    int *p = it.getKey<int>();
							    if( p && *p >= key )
							    {
							        key = *p + 1;
							    }
							}

							named->setEntry(key, t);
						}
						else
						{
							assert( m_g.m_dom.size() > 1 );
							object_table_ptr array_table(new ObjectTable());
							setXmlTags(array_table, elem_name, g_xml_type_array);
							array_table->setEntry(0,named);
							array_table->setEntry(1,t);
							top->setEntry(elem_name,array_table);
							m_g.m_dom.push(array_table);
						}

					}
					else
					{
						top->setEntry(elem_name,t);
					}
				}
				m_g.m_dom.push(t);
			}
		};

		struct action_etag_name : public action
		{
			action_etag_name(xml_grammar& g):action(g){}
			void operator()(const char* begin, const char* end) const
			{
				object_table_ptr top(boost::dynamic_pointer_cast<ObjectTable>(m_g.m_dom.top()));
				std::string top_type_value;
				std::string top_name;
				getXmlTags(top, top_name, top_type_value);

				if (!top->hasEntry(g_xml_type_value))
					top->setEntry(g_xml_type_value,"");
				std::string elem_name(begin, end-begin);

				assert( elem_name == top_name );
				m_g.m_dom.pop();
				top = boost::dynamic_pointer_cast<ObjectTable>(m_g.m_dom.top());
				getXmlTags(top, top_name, top_type_value);
				if ( top_name == elem_name )
				{
					if ( top_type_value == g_xml_type_array )
						m_g.m_dom.pop();
				}

			}
		};

		struct action_attrib_name : public action
		{
			action_attrib_name(xml_grammar& g):action(g){}
			void operator()(const char* begin, const char* end) const
			{
				assert( m_g.m_current_attrib_name.empty() );
				object_table_ptr top(boost::dynamic_pointer_cast<ObjectTable>(m_g.m_dom.top()));
				std::string top_name;
				std::string top_type_value;
				getXmlTags(top, top_name, top_type_value);
				assert( top_type_value == g_xml_type_element );

				m_g.m_current_attrib_name.assign(begin, end-begin);
			}
		};

		struct action_attrib_value : public action
		{
			action_attrib_value(xml_grammar& g):action(g){}
			void operator()(const char* begin, const char* end) const
			{
				assert( !m_g.m_current_attrib_name.empty() );
				std::string attrib_value(begin, end-begin);

				object_table_ptr top(boost::dynamic_pointer_cast<ObjectTable>(m_g.m_dom.top()));
				std::string top_name;
				std::string top_type_value;
				getXmlTags(top, top_name, top_type_value);
				assert( top_type_value == g_xml_type_element );

				object_table_ptr attribs;
				if ( !top->getEntry(g_xml_type_attrib,attribs) )
				{
					attribs.reset( new ObjectTable() );
					setXmlTags(attribs, g_xml_type_attrib, g_xml_type_attrib);
					top->setEntry(g_xml_type_attrib, attribs);
				}

				if ( attribs->hasEntry( m_g.m_current_attrib_name ) )
				{
					throw NapalmError(std::string("Already have attrib '")
						+ m_g.m_current_attrib_name + "' in element '"
						+ top_name + "'");
				}

				attribs->setEntry( m_g.m_current_attrib_name, attrib_value );
				m_g.m_current_attrib_name.clear();
			}
		};

		xml_grammar()
			:	m_a_prolog(*this)
			,	m_a_int(*this)
			,	m_a_float(*this)
			,	m_a_str(*this)
			,	m_a_stag_name(*this)
			,	m_a_etag_name(*this)
			,	m_a_attrib_name(*this)
			,	m_a_attrib_value(*this)
		{}

		std::stack<object_ptr> m_dom;
		std::string m_current_attrib_name;

		action_prolog m_a_prolog;
		action_int m_a_int;
		action_float m_a_float;
		action_str m_a_str;
		action_stag_name m_a_stag_name;
		action_etag_name m_a_etag_name;
		action_attrib_name m_a_attrib_name;
		action_attrib_value m_a_attrib_value;
	};


	object_table_ptr parseXml(const std::string& xml_str, std::string* perror )
	{

		// !!!
		// !!! Mutex :  Remove once  spirit is thread safe
		// !!!
		boost::mutex::scoped_lock l(g_parse_mtx);

		xml_grammar g;
		bs::parse_info<> info = bs::parse(xml_str.c_str(), g);

		if(!info.full)
		{
			std::ostringstream strm;
			int lineCount = 0;
			int charInLineCount = 0;
			for ( const char * p = xml_str.c_str() ; p != info.stop ; ++p )
			{
				if ( *p == '\n' )
				{
					++lineCount;
					charInLineCount = 0;
				}
				else
					++charInLineCount;
			}

			strm << std::string( xml_str.c_str(), info.stop ) << '\n';
			for ( int i = 0 ; i < charInLineCount ; ++i )
				strm << ' ';
			strm << "^\nParser error, line " << lineCount;

			if(perror)
			{
				*perror = strm.str();
				return object_table_ptr();
			}
			else
			{
				throw NapalmError(std::string("\n") + strm.str());
			}
		}


		assert(!g.m_dom.empty());
		object_table_ptr t = boost::dynamic_pointer_cast<ObjectTable>(g.m_dom.top());
		assert(t);
		assert(t->size()==3); // __name__, __type__ && 1 top-level element

#ifdef BOOST_SPIRIT_DEBUG
		object_rawptr_set printed;
		std::cout << "-----------------------------------" << std::endl;
		t->dump(std::cout, printed);
		std::cout << "-----------------------------------" << std::endl;
#endif

		return t;
	}

}
