#!/bin/sh

# Script for downloading dance music
# Written by Simon Dixon, based on a script by Guenther Geiger

# Usage examples:
#		getdance.sh Samba
#		getdance.sh "Viennese Waltz"
#		getdance.sh Rumba International

if [ $# == 0 ] ; then
	echo usage: $0 "<dance> [<style>]" 
	exit
fi
if [ ! -d tmp ] ; then
	mkdir tmp
fi

DANCE=$1
STYLE=$2
d=`echo $DANCE | sed 's/ //g'`
s=`echo $STYLE | sed 's/ //g'`
if test "$s" == "" ; then
	danceName=$d
else
	danceName=$d-$s
fi

# Get descriptive pages
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
	wget "http://www.ballroomdancers.com/Music/search_style.asp?PageNum=$i&Dance=$DANCE&Style=$STYLE" 
done

# Extract ram file URLs and metadata
>"$danceName".tmp
>"$danceName".log
for i in `ls search*`; do
	cat "$i" | tr ">" "\n" | grep ram | sed s/\<A\ HREF=\'// | sed s/\'// >> "$danceName".tmp
	cat "$i" | tr ">" "\n" | awk -f edit.awk >>"$danceName".log
	mv "$i" tmp/.
done
# grep ".ram\$" log1 | sort | uniq > $danceName.chk1
# sort $danceName.tmp > $danceName.chk2

# Get the link files and then the realaudio files
mkdir "$danceName"-ra
cd "$danceName"-ra
for i in `cat ../"$danceName".tmp`; do
	file1=`echo $i | cut -c38- | tr "/" "-"`
	file2=`basename $file1 .ram`.ra
	wget -O $file1 $i
	wget -i $file1 -O $file2
	mv $file1 ../tmp/.
	if file $file2 | grep -v RealMedia >>../"$danceName".log ; then
		mv $file2 ../tmp/.
	fi
done
cd ..
mv "$danceName".tmp tmp/.

echo FINISHED
echo "Note:  The tmp directory can be removed if everything went OK"
