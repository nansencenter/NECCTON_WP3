#!/bin/bash
set -u
shopt -s extglob # for using bash built-in string manipulation

if [[ ! $# -eq 2 ]] ; then
    echo "Usage: "
    echo "     bash daily2nc.sh year mode    "
    echo " ex) bash daily2nc.sh 2009 run/test"
    exit 1
fi

isleap() { 
   year=$1
   (( !(year % 4) && ( year % 100 || !(year % 400) ) )) &&
       return 0 || # true : leap year
       return 1    # false: not a leap
}

year=$1
mode=$2

if isleap $year; then
    echo "$year : Leap Year"
    njd=366
else
    echo "$year : Normal Year"
    njd=365
fi    

#-- common directories

HYCTOOL=/cluster/home/wakamatsut/TP2/NERSC-HYCOM-CICE/hycom/MSCPROGS/bin              # betzy
#HYCTOOL=/cluster/home/wakamatsut/BIORAN_202301/bioran/hycom_r525_BIORAN/MSCPROGS/bin  # fram
CNFGDIR=$PWD
TOOLDIR=$PWD
ARCHDIR=/nird/datalake/NS9481K/shuang/TP2_output/expt_04.2
DATADIR=$PWD/data

mkdir -p $DATADIR 

#-- Source and Archive

if [ ! -d $ARCHDIR ]; then
    echo "Can not find archive folder, STOP"
    echo "ARCDIR:$ARCDIR"
    exit
fi   
    
#-- link/calculate daily mean

pushd $DATADIR >/dev/null 2>&1

SRCDIR=$CNFGDIR

ln -sf $SRCDIR/regional.depth.a .
ln -sf $SRCDIR/regional.depth.b .
ln -sf $SRCDIR/regional.grid.a .
ln -sf $SRCDIR/regional.grid.b .
ln -sf $SRCDIR/TP2depth.nc archm_depth.nc

SRCDIR=$TOOLDIR

ln -sf $SRCDIR/extract.archm .

SRCDIR=$HYCTOOL

export PATH=$PATH:$HYCTOOL

for jd in $(seq 1 $njd)
do    

doy=`printf "%03d\n" $jd`                  # day of the year [DDD] eg) 20070103 => 003

fhead=archm.${year}_${doy}_12 # TP2 daily file

afile=${fhead}.a
bfile=${fhead}.b

ncfile=archm.${year}_12.nc

if [ -f $ARCHDIR/${afile} ]; then
    ln -sf $ARCHDIR/${afile} .
    ln -sf $ARCHDIR/${bfile} .
    file[$jd]=$afile
    echo "  $afile added to file list";
else
    echo "  $afile does not exist, SKIP";
fi

done

# create daily mean annual file (netcdf) for the given year

if [ "$mode" == "run" ]; then

    if [ ! -f $ncfile ]; then
       h2nc ${file[@]};
       mv tmp1.nc $ncfile
    else
       echo "$ncfile exists, SKIP"
    fi

    # remove simlinks

    for file in archm*.a
    do
       rm $file
    done

    for file in archm*.b
    do
       rm $file
    done

    # prepare "normal year" files

    nyfile="${ncfile%.nc}_ny.nc"

    if isleap $year; then
      # remove leap day if leap year.
      ncks -F -d rdim,1,59 -d rdim,61,366 $ncfile $nyfile
    else
      ln -sf $ncfile  $nyfile
    fi
fi    

popd >/dev/null 2>&1

exit
