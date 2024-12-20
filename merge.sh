#!/bin/bash

# 输出文件名
output="merged.cu"

# 清空输出文件
> $output

base="CUDA-NN/src"
include_folder="CUDA-NN/include"

if [ -z "$1" ]; then
  echo "Usage: $0 <main|test>"
  exit 1
fi

if [ "$1" == "main" ]; then
  main_file="$base/main.cu"
elif [ "$1" == "test" ]; then
  main_file="$base/test.cu"
else
  echo "Invalid argument: $1. Use 'main' or 'test'."
  exit 1
fi

should_add_line() {
    local line="$1"

    # 判断条件，返回0表示可以添加，返回1表示不可以添加
    if [[ $line =~ ^[[:space:]]*#include[[:space:]]*\<.*\> ]]; then
        line=$(echo "$line" | sed 's/#include[[:space:]]*#include/#include/' | sed 's/[[:space:]]*//g')
        if [[ -z "${include_set[$line]}" ]]; then
            includes+="$line"$'\n'  # 保存尖括号 include 语句
            include_set[$line]=1    # 标记为已添加
        fi
        return 1
    # 跳过双引号 include 语句
    elif [[ $line =~ \#include[[:space:]]*\".*\" ]]; then
        return 1
    elif [[ $line == \#define\ * && $line != *[\(\)\[\]\{\}\\=]* && ! $line =~ \#define\ [a-zA-Z_][a-zA-Z0-9_]*\ [-+]?[0-9]*\.?[0-9]*([eE][-+]?[0-9]+)? ]]; then
        return 1
    elif [[ $line == \#ifndef\ * && $line != *[\(\)\[\]\{\}\\=]* && ! $line =~ \#define\ [a-zA-Z_][a-zA-Z0-9_]*\ [0-9]+ ]]; then
        return 1
    elif [[ $line == \#endif\ \/* && ! $line =~ *[[:digit:]]* ]]; then
        return 1
    elif [[ $line == \#pragma* ]]; then
        return 1
    fi

    return 0  # 默认可以添加
}



# 定义合并的顺序，将 kernels 文件夹放在最前面
folders=("tensor/kernels" "common" "tensor" "layers" "models" "datasets")

# 使用关联数组来指定特定文件夹及其对应的编译顺序
declare -A specific_folders
specific_folders=(
    ["models"]="base stn3d stnkd encoder pointnet"
    ["layers"]="module base"  # 添加更多的文件夹及其顺序
)

includes=""
declare -A include_set

# 合并 include 文件夹中的 .cuh 文件
if [ -d "$include_folder" ]; then
    echo "Merging .cuh files from $include_folder"
    
    for file in $(find $include_folder -maxdepth 1 -name "*.cuh" | sort); do
        echo "// Merging: $file" >> $output
        last_line=""
        while IFS= read -r line || [ -n "$line" ]; do
            should_add_line "$line"
            if [[ $? -eq 0 ]]; then
                echo "$line" >> $output  # 添加行到输出文件
            fi
            last_line="$line"
        done < "$file"
        
        echo -e "\n" >> $output  # 每个文件末尾加一个空行
    done
fi

# 遍历文件夹，按照指定顺序合并 .cu 文件
for folder in "${folders[@]}"; do
    merged_files=()
    echo "Merging folder: $folder"

    if [[ -v specific_folders[$folder] ]]; then
        # 如果当前文件夹是 specific_folders，则按照指定顺序合并文件
        for file_prefix in ${specific_folders[$folder]}; do
            for ext in "cuh" "cu"; do
                file="$base/$folder/$file_prefix.$ext"
                if [[ -f "$file" ]]; then
                    echo "// Merging: $file" >> $output
                    while IFS= read -r line || [ -n "$line" ]; do
                        should_add_line "$line"
                        if [[ $? -eq 0 ]]; then
                            echo "$line" >> $output
                        fi
                    done < "$file"
                    echo -e "\n" >> $output  
                    merged_files+=("$file")
                fi
            done
        done
    fi

    # 处理当前文件夹中其他的 .cuh 文件
    for file in $(find $base/$folder -maxdepth 1 -name "*.cuh" | sort); do
        if [[ ! " ${merged_files[*]} " =~ " $file " ]]; then
            echo "// Merging: $file" >> $output
            last_line=""

            # 处理 .cuh 文件，提取尖括号 include 语句，并删除双引号 include 和 #ifndef, #define, #endif
            while IFS= read -r line || [ -n "$line" ]; do
                should_add_line "$line"
                if [[ $? -eq 0 ]]; then
                    echo "$line" >> $output  # 添加行到输出文件
                fi
                last_line="$line"
            done < "$file"
            echo -e "\n" >> $output 
        fi
    done

    for file in $(find $base/$folder -maxdepth 1 -name "*.cu" | sort); do
        if [[ ! " ${merged_files[*]} " =~ " $file " ]]; then
            echo "// Merging: $file" >> $output
            last_line=""
            
            while IFS= read -r line || [ -n "$line" ]; do
                should_add_line "$line"
                if [[ $? -eq 0 ]]; then
                    echo "$line" >> $output
                fi
                last_line="$line"
            done < "$file"

            echo -e "\n" >> $output
        fi
    done

done


echo "// Merging: $main_file" >> $output
while IFS= read -r line || [ -n "$line" ]; do
    should_add_line "$line"
    if [[ $? -eq 0 ]]; then
        echo "$line" >> $output  # 添加行到输出文件
    fi
done < "$main_file"
echo -e "\n" >> $output


temp_file=$(mktemp)
echo -e "$includes" > $temp_file
cat "$output" >> "$temp_file"
mv "$temp_file" "$output"

echo "All files merged into $output"
