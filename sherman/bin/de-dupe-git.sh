for f in $(cat sherfiles); do
DIR=split.$f

for stub in $(cat stubs); do
STUBFILE=$(echo $stub | perl -e'$foo = <STDIN>; $foo =~ s/\//-/g; $foo =~ s/-$//; print $foo')
if [ -e $DIR/$STUBFILE ]; then

echo move $DIR/$STUBFILE to repo
mv -f $DIR/$STUBFILE ./testgit/$STUBFILE &&
echo commit ./testgit/$STUBFILE &&
cd testgit &&
git commit -m "$f" ./$STUBFILE
cd .. &&
echo move repo file back to ./testgit/$STUBFILE &&
mv -f ./testgit/$STUBFILE $DIR/$STUBFILE &&

echo done
fi
done

done

