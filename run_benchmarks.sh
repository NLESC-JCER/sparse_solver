#!/bin/bash
CWD="$(pwd)"
for thread in 1 2 4
do 
echo running on ${thread} 
logfolder=${thread}_threads
mkdir $logfolder
	for folder in seq*
	do
	echo $folder
	cd $folder
	for matrix in A_mat_eigen_*
		do 
		number=$( echo $matrix | awk -F'[_.]' '{print $5}' )
		echo $number
		b_vec="b_vec_MUMPS_${number}".txt
		~/Programs/sparse_solver/build/src/benchmark -A $matrix -b $b_vec -t $thread -s 1e-14 > ${CWD}/${logfolder}/${folder}_${number}.log
		done
	cd ..
	done
done
