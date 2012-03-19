#include <boost/spirit.hpp>
#include <stack>
#include <sstream>
#include "parsePythonDict.h"
#include "napalm/core/exceptions.h"



namespace bs = boost::spirit;

namespace napalm {


// grammar for python dict parsing
struct py_dict_grammar : public bs::grammar<py_dict_grammar>
{
	template<class ScannerT>
	struct definition
	{
		bs::rule<ScannerT>
			quoted_str,
			boolean,
			key,
			value,
			table_entry,
			table,
			list_entry,
			list,
			expression;

		definition(const py_dict_grammar& g)
		{
			boolean = bs::str_p("True") | bs::str_p("False");
			quoted_str = bs::lexeme_d['\'' >> *(bs::anychar_p - '\'') >> '\''] |
					bs::lexeme_d['"' >> *(bs::anychar_p - '"') >> '"'];
			table_entry = key >> ':' >> value;
			list_entry = value[g.m_action_list_val];
			expression = table | list;

			key =
				bs::int_p			[g.m_action_int_key]
			    | quoted_str		[g.m_action_str_key];

			value =
				bs::strict_real_p	[g.m_action_float_val]
				| bs::int_p			[g.m_action_int_val]
				| boolean			[g.m_action_bool_val]
				| quoted_str		[g.m_action_str_val]
				| expression;

			table = (bs::ch_p('{')[g.m_action_table_begin]
			    >> !(table_entry >> *(',' >> table_entry)) >> '}')[g.m_action_table];

			list = (bs::ch_p('[')[g.m_action_table_begin]
			    >> !(list_entry >> *(',' >> list_entry)) >> ']')[g.m_action_table];

			BOOST_SPIRIT_DEBUG_RULE(quoted_str);
			BOOST_SPIRIT_DEBUG_RULE(boolean);
			BOOST_SPIRIT_DEBUG_RULE(key);
			BOOST_SPIRIT_DEBUG_RULE(value);
			BOOST_SPIRIT_DEBUG_RULE(table_entry);
			BOOST_SPIRIT_DEBUG_RULE(table);
			BOOST_SPIRIT_DEBUG_RULE(list_entry);
			BOOST_SPIRIT_DEBUG_RULE(list);
			BOOST_SPIRIT_DEBUG_RULE(expression);
		}

		bs::rule<ScannerT> const &start() { return expression; }
	};

	struct action {
		py_dict_grammar& m_g;
		action(py_dict_grammar& g):m_g(g){}
	};

	struct action_int_key : public action {
		action_int_key(py_dict_grammar& g):action(g){}
		void operator()(int val) const {
			m_g.m_keys.push( val );
			++(m_g.m_entryCount.top());
		}
	};

	struct action_str_key : public action {
		action_str_key(py_dict_grammar& g):action(g){}
		void operator()(const char* begin, const char* end) const {
			m_g.m_keys.push( std::string(begin+1, end-begin-2) );
			++(m_g.m_entryCount.top());
		}
	};

	struct action_int_val : public action {
		action_int_val(py_dict_grammar& g):action(g){}
		void operator()(int val) const {
			m_g.m_values.push(object_ptr(new IntAttrib(val)));
		}
	};

	struct action_bool_val : public action {
		action_bool_val(py_dict_grammar& g):action(g){}
		void operator()(const char* begin, const char* end) const {
			m_g.m_values.push(object_ptr(new BoolAttrib((*begin=='T')? true : false)));
		}
	};

	struct action_float_val : public action {
		action_float_val(py_dict_grammar& g):action(g){}
		void operator()(float val) const {
			m_g.m_values.push(object_ptr(new FloatAttrib(val)));
		}
	};

	struct action_str_val : public action {
		action_str_val(py_dict_grammar& g):action(g){}
		void operator()(const char* begin, const char* end) const {
			std::string s(begin+1, end-begin-2);
			m_g.m_values.push(object_ptr(new StringAttrib(s)));
		}
	};

	struct action_list_val : public action {
		action_list_val(py_dict_grammar& g):action(g){}
		void operator()(const char*,const char*) const {
			m_g.m_keys.push(m_g.m_entryCount.top() );
			++(m_g.m_entryCount.top());
		}
	};

	struct action_table_begin : public action {
		action_table_begin(py_dict_grammar& g):action(g){}
		void operator()(char) const {
			m_g.m_entryCount.push(0);
		}
	};

	struct action_table : public action {
		action_table(py_dict_grammar& g):action(g){}
		void operator()(const char* begin, const char* end) const
		{
			assert(!(m_g.m_entryCount.empty()));

			unsigned int nentries = m_g.m_entryCount.top();
			assert(m_g.m_keys.size() >= nentries);

			object_table_ptr t(new ObjectTable());
			for(unsigned int i=0; i<nentries; ++i)
			{
				t->setEntry(m_g.m_keys.top(), m_g.m_values.top());
				m_g.m_keys.pop();
				m_g.m_values.pop();
			}

			m_g.m_values.push(t);
			m_g.m_entryCount.pop();
		}
	};

	py_dict_grammar()
	:	m_action_int_key(*this),
		m_action_str_key(*this),
		m_action_int_val(*this),
		m_action_bool_val(*this),
		m_action_float_val(*this),
		m_action_str_val(*this),
		m_action_list_val(*this),
		m_action_table_begin(*this),
		m_action_table(*this)
	{}

	std::stack<int> m_entryCount;
	std::stack<TableKey> m_keys;
	std::stack<object_ptr> m_values;

	action_int_key 		m_action_int_key;
	action_str_key 		m_action_str_key;
	action_int_val 		m_action_int_val;
	action_bool_val 	m_action_bool_val;
	action_float_val 	m_action_float_val;
	action_str_val 		m_action_str_val;
	action_list_val		m_action_list_val;
	action_table_begin	m_action_table_begin;
	action_table		m_action_table;
};


object_table_ptr parsePythonDict(const std::string& dict_str, std::string* perror)
{

	// !!!
	// !!! Mutex :  Remove once  spirit is thread safe
	// !!!
	boost::mutex::scoped_lock l(g_parse_mtx);


	py_dict_grammar g;
	std::string dict_str_ = pystring::strip(dict_str);
	bs::parse_info<> info = bs::parse(dict_str_.c_str(), g, bs::space_p);

	// todo once there are >1 parsers, move this into common func and deal with clamping
	// long string, multi-line etc
	if(!info.full)
	{
		std::ostringstream strm;
		strm << dict_str_ << '\n';
		for(const char* p=dict_str_.c_str(); p!=info.stop; ++p)
			strm << ' ';
		strm << "^\nParser error.";

		if(perror)
		{
			*perror = strm.str();
			return object_table_ptr();
		}
		else
			throw NapalmError(std::string("\n") + strm.str());
	}

	assert(g.m_keys.empty());
	assert(g.m_entryCount.empty());
	assert(g.m_values.size() == 1);

	object_table_ptr t = boost::dynamic_pointer_cast<ObjectTable>(g.m_values.top());
	assert(t);
	return t;

}


}


