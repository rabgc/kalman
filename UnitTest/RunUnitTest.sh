if [ -s ./unit_test.e ]; then 
   rm ./unit_test.e
fi

make

if [ -s unit_test.e ]; then
   ./unit_test.e -d -s | sed -e "/with expans/,+1d"
else
   echo "could not find ./unit_test.e" 
fi
