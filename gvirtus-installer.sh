#! /bin/bash

sdk_version=`cat $CUDA_HOME/version.txt|awk '{ print $3 }'|tr '.' ' '|awk '{ print $1 "." $2 }'`
CUDNN_MAJOR=`grep CUDNN_MAJOR $CUDA_HOME/include/cudnn.h | head -n 1| awk '{ print $3 }'`
CUDNN_MINOR=`grep CUDNN_MINOR $CUDA_HOME/include/cudnn.h | head -n 1| awk '{ print $3 }'`
CUDNN_PATCHLEVEL=`grep CUDNN_PATCHLEVEL $CUDA_HOME/include/cudnn.h | head -n 1| awk '{ print $3 }'`

if [ $# -ne 1 ]; then
  /bin/echo -e "\e[1;91mUsage: $0 \"path-of-installation-folder\"\e[0m"
  exit 1
fi

INSTALL_FOLDER=$1

for module in $(find . -maxdepth 1 -type d -regextype egrep -regex "./gvirtus(\.[a-zA-Z]*|\b)" | sort); do
  cd $module
  bash install.sh ${INSTALL_FOLDER}
  cd $OLDPWD
done

mkdir $INSTALL_FOLDER/etc
cat > $INSTALL_FOLDER/etc/profile-frontend << EOF
#! /bin/bash
export GVIRTUS_HOME=$INSTALL_FOLDER
export LD_LIBRARY_PATH=\$GVIRTUS_HOME/external/lib:\$GVIRTUS_HOME/lib/frontend:\$GVIRTUS_HOME/lib:\$GVIRTUS_HOME/lib/communicator:/usr/local/lib64:\$LD_LIBRARY_PATH
export EXTRA_NVCCFLAGS="--cudart=shared"
export DEFAULT_ENDPOINT=0
EOF

cat > $INSTALL_FOLDER/etc/profile-backend << EOF
#!/bin/bash
export GVIRTUS_HOME=$INSTALL_FOLDER
export CUDA_HOME=$CUDA_HOME
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$GVIRTUS_HOME/lib:\$GVIRTUS_HOME/lib/communicator:\$GVIRTUS_HOME/lib/backend:\$GVIRTUS_HOME/external/lib:\$LD_LIBRARY_PATH
export PATH=\$CUDA_HOME/bin:\$PATH
export EXTRA_NVCCFLAGS="--cudart=shared"
EOF


lpath=${INSTALL_FOLDER}/lib/frontend
for lib in $(find $lpath -name "*.so" | sed 's|^./||'); do
  libName=`basename $lib`
  echo $libName
  if [[ "$libName" == "libcudnn.so" ]]; then
    ln -sf $lib $lib.$CUDNN_MAJOR
    ln -sf $lib $lib.$CUDNN_MAJOR.$CUDNN_MINOR
    ln -sf $lib $lib.$CUDNN_MAJOR.$CUDNN_MINOR.$CUDNN_PATCHLEVEL
    ln -sf $lib $lib.$sdk_version
  else
    ln -sf $lib $lib.$sdk_version

  fi
done
/bin/echo -e "\e[1;30;102mSYMBOLIC LINKS TO LIBRARY CREATED!\e[0m"
echo

mkdir $INSTALL_FOLDER/dist
tar -cvzf $INSTALL_FOLDER/dist/gvirtus-frontend.tar.gz $INSTALL_FOLDER/etc/properties.json $INSTALL_FOLDER/etc/profile-frontend $INSTALL_FOLDER/lib/communicator/* $INSTALL_FOLDER/lib/frontend/* $INSTALL_FOLDER/lib/libgvirtus-frontend.so


/bin/echo -ne "\e[1;97;44;5mDo you want clean build files? (y|n) [Default: n]:\e[0m  "
read flag
if [[ $flag == "y" || $flag == "yes" ]]; then
  echo
  echo "-- Build files will be removed!"
  echo "$(bash ./gvirtus-cleaner.sh)"
else
  echo
  echo "-- Build files will not be removed!"
  /bin/echo -e "-- If you want clean build files run \e[1;92m.$PWD/gvirtus-cleaner.sh\e[0m"
fi

echo
/bin/echo -e "\e[1;97;42mINSTALLATION COMPLETED!\e[0m"
echo
