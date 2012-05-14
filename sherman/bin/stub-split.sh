#for f in $(tac sherfiles); do
for f in $(find . -type d -name split\*); do

DIR=$f
FILE=$DIR/sorted

#mkdir $DIR

if [ ! -e $FILE ]; then
echo make $DIR and sort to $FILE
sort -k 4 $f > $FILE
rm -f $f
fi

for stub in $(cat stubs); do

STUBFILE=$DIR/$(echo $stub | perl -e'$foo = <STDIN>; $foo =~ s/\//-/g; $foo =~ s/-$//; print $foo')
if [ ! -e $STUBFILE ]; then
#echo grep $FILE for $stub into $STUBFILE
TABCOUNT=$(tail -1 $FILE | awk '{print gsub(/\t/,"")+1}')
echo "grep --mmap -P \"\\\t$stub\" $FILE | sort -k $TABCOUNT > $STUBFILE"
fi

done

done
