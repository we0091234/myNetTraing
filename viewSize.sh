#!/bin/bash
path1="$1*"
echo "$path1"
du -s $path1  | sort -rn | awk '{split($2,a,"/");if($1>=1024&&$1<1024**2)\
{printf "%-10.2fM\t\t%s\n",$1/1024,a[length(a)]}\
else if($1>=1024**2){printf "%-10.2fG\t\t%s\n",$1/1024/1024,a[length(a)]}\
else{printf"%-10.2fK\t\t%s\n", $1"K",a[length(a)]}}'
