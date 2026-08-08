import os
import subprocess
import numpy as np
import shutil
import glob
import time
# cd /home/yyShen/NAcontact/data/affinity_data/cg_input/

data_type = "c-met_AF3"
pair_list = np.loadtxt(f'/home/yyShen/NAcontact/data/PPB-Affinity/{data_type}_pair.txt', dtype=str)
# pair_list = np.loadtxt(f'/home/yyShen/NAcontact/data/PPB-Affinity/tcr_bu.txt', dtype=str)
# pair_list = np.loadtxt(f'check_pairs_{data_type}.txt', dtype=str)
# base_dir = f"/home/yyShen/NAcontact/data/affinity_data/cg_input/cg_{data_type}/"
cg_input_dir = "/home/yyShen/NAcontact/data/affinity_data/cg_input/"
base_dir = f"{cg_input_dir}cg_{data_type}/"
check_file = f"pairs_{data_type}.txt"

with open(check_file, "a") as ft:
    for pair in pair_list:
        # pair = '6ysq_AC_G'
        pair_dir = os.path.join(base_dir, pair)
        os.chdir(pair_dir)

        # === Step 1: 检查是否已完成 ===
        em_gro = os.path.join(pair_dir, "em.gro")
        em_tpr = os.path.join(pair_dir, "em.tpr")
        if (os.path.isfile(em_gro) and os.path.getsize(em_gro) > 0 and
                os.path.isfile(em_tpr) and os.path.getsize(em_tpr) > 0):
            # print(f"{pair} em.gro and em.tpr exist, skip")
            continue
        else:
            ft.write(f"{pair}\n")

            print(f"{pair} new generation")
            # 拷贝 .itp 文件到当前工作目录
            itp_files = ["martini_v2.2.itp", "martini_v2.0_ions.itp",
                         "min_steep.mdp", "min_cg.mdp", "water.gro", "minim.mdp"]
            for itp_file in itp_files:
                src = os.path.join(cg_input_dir, itp_file)
                if os.path.exists(src):
                    shutil.copy(src, ".")
                    print(f"Copied {itp_file} to {pair_dir}")
                else:
                    print(f"{itp_file} not found")

            # 修改 cg_M2.top 文件
            top_file = "cg_M2.top"
            if not os.path.exists(top_file):
                print(f"{top_file} not exist, skip {pair}")
                continue

            with open(top_file, "r") as f:
                lines = f.readlines()

            # 替换 include 语句
            new_lines = []
            include_replaced = False
            for line in lines:
                if '#include "martini.itp"' in line:
                    new_lines.append(f'#include "martini_v2.2.itp"\n')
                    new_lines.append(f'#include "martini_v2.0_ions.itp"\n')
                    include_replaced = True
                else:
                    new_lines.append(line)

            # # 确保 [ molecules ] 章节末尾有换行
            # if "[ molecules ]" in "".join(new_lines):
            #     if not new_lines[-1].endswith("\n"):
            #         new_lines.append("\n")

            # 修复 [ molecules ] 部分，仅保留 Protein_* 分子定义
            in_mol_section = False
            final_lines = []
            for line in new_lines:
                stripped = line.strip()

                # 检测到 [ molecules ]，进入处理阶段
                if stripped.startswith("[ molecules ]"):
                    in_mol_section = True
                    final_lines.append(line)
                    continue

                if in_mol_section:
                    if stripped == "" or stripped.startswith(";"):
                        final_lines.append(line)
                        continue
                    elif stripped.startswith("["):  # 遇到下一个 section，退出 molecules 处理
                        in_mol_section = False
                        final_lines.append(line)
                        continue
                    else:
                        parts = stripped.split()
                        if len(parts) == 2:
                            name, count = parts
                            if name.startswith("Protein"):
                                final_lines.append(f"{name}\t{count}\n")
                        # 其他（如 W、NA+、CL-）将被自动丢弃
                else:
                    final_lines.append(line)

            # 写回修改后的 top 文件
            with open(top_file, "w") as f:
                f.writelines(final_lines)

            if not include_replaced:
                print(f"Warning: 'martini.itp' not found in {top_file}!")

            # **运行 GROMACS 命令**
            commands = [
                "gmx editconf -f cg_M2.pdb -o complex_box.gro -c -d 1.2 -bt triclinic",  # 1.2,1.8,2.0,2.2
                "gmx solvate -cp complex_box.gro -cs water.gro -p cg_M2.top -o protein_sol.gro -radius 0.25",
                "gmx grompp -f minim.mdp -c protein_sol.gro -p cg_M2.top -o ions.tpr -maxwarn 1",
                "echo 13 | gmx genion -s ions.tpr -p cg_M2.top -o protein_ion.gro -pname NA+ -nname CL- -neutral -seed 3407",

                # Step 1: 最速下降法 (Steepest Descent)
                "gmx grompp -f min_steep.mdp -c protein_ion.gro -p cg_M2.top -o min_steep.tpr -maxwarn 2",
                "gmx mdrun -deffnm min_steep -nt 1 > en_min_steep.log 2>&1",

                # Step 2: 共轭梯度法 (Conjugate Gradient)
                "gmx grompp -f min_cg.mdp -c min_steep.gro -p cg_M2.top -o em.tpr -maxwarn 2",
                "nohup gmx mdrun -deffnm em -nt 1 > en_min.log 2>&1 &",

                # "nohup gmx mdrun -deffnm em -ntmpi 8 -ntomp 6 > enmin.log 2>&1 &"
            ]

            for cmd in commands:
                print(f"Running: {cmd}")
                subprocess.run(cmd, shell=True, check=True)

            print(f"✅ energy min finished！{pair}")
            ft.write(f"{pair}\n")

            # 等待所有 GROMACS 进程完成
            while True:
                result = subprocess.run("pgrep -u $USER mdrun", shell=True, stdout=subprocess.PIPE)
                if not result.stdout:
                    print("✅ GROMACS finished, start delete files!")
                    break
                else:
                    print("⏳ GROMACS still running, waiting 5 seconds...")
                    time.sleep(5)

            # 删除以 #em*# 开头的文件
            files_to_delete = glob.glob("#*#")+glob.glob("dd_dump_err*")+glob.glob("step*")
            print(files_to_delete)
            for file in files_to_delete:
                try:
                    os.remove(file)
                    print(f"Deleted file: {file}")
                except Exception as e:
                    print(f"Error deleting file {file}: {e}")

        # # 检查 em.gro 和 em.tpr 是否存在且非空
        # em_gro_exists = os.path.isfile("em.gro") and os.path.getsize("em.gro") > 0
        # em_tpr_exists = os.path.isfile("em.tpr") and os.path.getsize("em.tpr") > 0
        #
        # if not (em_gro_exists and em_tpr_exists):
        #     ft.write(f"{pair}\n")
