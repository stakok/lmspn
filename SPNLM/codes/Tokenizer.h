/* 
 * File:   Tokenizer.h
 * Author: weichen_cheng
 *
 * Created on November 18, 2013, 10:09 AM
 */

#ifndef TOKENIZER_H
#define	TOKENIZER_H

#include <functional>
#include <string>
#include <vector>
#include <algorithm>
#include <locale>

using namespace std;

//For the case the default is a space.
//This is the default predicate for the Tokenize() function.
class CIsSpace : public unary_function<char, bool>
{
public:
	bool operator()(char c) const;
};

inline bool CIsSpace::operator()(char c) const
{
	//isspace<char> returns true if c is a white-space character (0x09-0x0D or 0x20)
	//return isspace<char>(c);
	return isspace(c);
}

//For the case the separator is a comma
class CIsComma : public unary_function<char, bool>
{
public:
	bool operator()(char c) const;
};

inline bool CIsComma::operator()(char c) const
{
	return (',' == c);
}

//For the case the separator is a character from a set of characters given in a string
class CIsFromString : public unary_function<char, bool>
{
public:
	//Constructor specifying the separators
	CIsFromString(string const& rostr) : m_ostr(rostr) {}
	bool operator()(char c) const;

private:
	string m_ostr;
};

inline bool CIsFromString::operator()(char c) const
{
	int iFind = m_ostr.find(c);
	if(iFind != string::npos)
		return true;
	else
		return false;
}

//String Tokenizer
template <class Pred=CIsSpace>
class CTokenizer
{
public:
	//The predicate should evaluate to true when applied to a separator.
	static void Tokenize(vector<string>& roResult, string const& rostr, Pred const& roPred=Pred());
};

//The predicate should evaluate to true when applied to a separator.
template <class Pred>
inline void CTokenizer<Pred>::Tokenize(vector<string>& roResult, string const& rostr, Pred const& roPred)
{
	//First clear the results vector
	roResult.clear();
	string::const_iterator it = rostr.begin();
	string::const_iterator itTokenEnd = rostr.begin();
	while(it != rostr.end())
	{
		//Eat seperators
		while(roPred(*it))
			it++;
		//Find next token
		itTokenEnd = find_if(it, rostr.end(), roPred);
		//Append token to result
		if(it < itTokenEnd)
			roResult.push_back(string(it, itTokenEnd));
		it = itTokenEnd;
	}
}

#endif	/* TOKENIZER_H */

