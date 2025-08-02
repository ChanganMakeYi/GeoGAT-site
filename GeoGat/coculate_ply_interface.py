import os
import subprocess
import argparse
import numpy as np
import meshio
from scipy.interpolate import RegularGridInterpolator
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa
import shutil
import freesasa
import tempfile
import pandas as pd
import math
import logging

# 设置日志
logging.basicConfig(
    filename="generate_ply.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Kyte-Doolittle 亲疏水性评分表（扩展支持RNA核苷酸）
hydrophobicity_scale = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2,
    'A': -0.5, 'C': -1.0, 'G': -0.7, 'U': -1.2  # RNA核苷酸亲疏水性（近似值）
}

def check_command(command, cmd_path=None):
    """检查命令是否在 PATH 中"""
    cmd = cmd_path if cmd_path else command
    if not shutil.which(cmd):
        raise FileNotFoundError(
            f"'{cmd}' 未找到。请确保安装并添加到 PATH。\n"
            f"安装建议：\n"
            f"- APBS: 'brew install apbs' 或 'conda install -c conda-forge apbs'\n"
            f"- PDB2PQR: 'pip install pdb2pqr'\n"
            f"- HBPLUS: 下载自 http://www.ebi.ac.uk/thornton-srv/software/HBPLUS/\n"
            f"- MSMS: 下载自 http://mgltools.scripps.edu/downloads/tars/releases/MSMSRELEASE/REL2.6.1/\n"
            f"- pdb_to_xyzr: 确保 MSMS 工具包中的 pdb_to_xyzr 可执行文件已安装"
        )
    return cmd

def create_directories(base_dir):
    """创建所需的子文件夹"""
    folders = ["pdb", "pqr", "dx", "hbplus", "msms", "ply"]
    for folder in folders:
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

def get_file_paths(input_pdb, output_dir):
    """生成文件路径"""
    base_name = os.path.splitext(os.path.basename(input_pdb))[0]
    create_directories(output_dir)
    return {
        "clean_pdb": os.path.join(output_dir, "pdb", f"{base_name}_clean.pdb"),
        "pqr": os.path.join(output_dir, "pqr", f"{base_name}.pqr"),
        "pqr_pdb": os.path.join(output_dir, "pdb", f"{base_name}_pqr.pdb"),
        "apbs_in": os.path.join(output_dir, "pqr", f"{base_name}_apbs.in"),
        "dx": os.path.join(output_dir, "dx", f"{base_name}.dx"),
        "hbplus_out": os.path.join(output_dir, "hbplus", f"{base_name}_hbplus.out"),
        "xyzr": os.path.join(output_dir, "msms", f"{base_name}.xyzr"),
        "vert": os.path.join(output_dir, "msms", f"{base_name}.vert"),
        "face": os.path.join(output_dir, "msms", f"{base_name}.face"),
        "ply": os.path.join(output_dir, "msms", f"{base_name}.ply"),
        "output_ply": os.path.join(output_dir, "ply", f"{base_name}_updated.ply"),
        "interface_csv": os.path.join(output_dir, f"interface_atoms.csv"),
        "protein_pdb": os.path.join(output_dir, "pdb", f"{base_name}_protein.pdb"),
        "rna_pdb": os.path.join(output_dir, "pdb", f"{base_name}_rna.pdb"),
        "protein_pqr": os.path.join(output_dir, "pqr", f"{base_name}_protein.pqr"),
        "rna_pqr": os.path.join(output_dir, "pqr", f"{base_name}_rna.pqr"),
        "protein_pqr_pdb": os.path.join(output_dir, "pdb", f"{base_name}_protein_pqr.pdb"),
        "rna_pqr_pdb": os.path.join(output_dir, "pdb", f"{base_name}_rna_pqr.pdb"),
        "protein_apbs_in": os.path.join(output_dir, "pqr", f"{base_name}_protein_apbs.in"),
        "rna_apbs_in": os.path.join(output_dir, "pqr", f"{base_name}_rna_apbs.in"),
    }

class ProteinSelect(Select):
    """选择蛋白质和RNA残基，排除水分子"""
    def accept_residue(self, residue):
        return residue.get_resname() != "HOH"

class ChainSelect(Select):
    """选择特定链（蛋白质或RNA）"""
    def __init__(self, chain_ids):
        self.chain_ids = chain_ids

    def accept_chain(self, chain):
        return chain.id in self.chain_ids

    def accept_residue(self, residue):
        return residue.get_resname() != "HOH"

def get_protein_rna_chains(structure):
    """区分蛋白质和RNA链"""
    protein_chains = []
    rna_chains = []
    for model in structure:
        for chain in model:
            residues = list(chain)
            if any(is_aa(residue, standard=True) for residue in residues):
                protein_chains.append(chain.id)
            elif any(residue.get_resname() in ["A", "C", "G", "U"] for residue in residues):
                rna_chains.append(chain.id)
    return protein_chains, rna_chains

def clean_pdb_file(input_pdb, output_pdb):
    """清理PDB文件，保留蛋白质和RNA"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", input_pdb)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, ProteinSelect())
    print(f"已生成清理后的PDB文件：{output_pdb}")
    logging.info(f"已生成清理后的PDB文件：{output_pdb}")

def split_chains(input_pdb, protein_pdb, rna_pdb, protein_chains, rna_chains):
    """分离蛋白质和RNA链"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", input_pdb)
    io = PDBIO()

    io.set_structure(structure)
    io.save(protein_pdb, ChainSelect(protein_chains))
    print(f"已生成蛋白质PDB文件：{protein_pdb}")
    logging.info(f"已生成蛋白质PDB文件：{protein_pdb}")

    io.set_structure(structure)
    io.save(rna_pdb, ChainSelect(rna_chains))
    print(f"已生成RNA PDB文件：{rna_pdb}")
    logging.info(f"已生成RNA PDB文件：{rna_pdb}")

def convert_pqr_to_pdb(pqr_file, pdb_file):
    """将PQR转换为PDB"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pqr_file)
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_file)
    print(f"已将PQR转换为PDB：{pdb_file}")
    logging.info(f"已将PQR转换为PDB：{pdb_file}")

def run_pdb2pqr(input_pdb, output_pqr, apbs_input):
    """运行PDB2PQR"""
    temp_input_pdb = tempfile.NamedTemporaryFile(delete=False, suffix='.pdb', mode='w', encoding='ascii').name
    with open(input_pdb, 'r', encoding='utf-8') as f:
        with open(temp_input_pdb, 'w', encoding='ascii') as tf:
            tf.write(f.read())

    cmd = ["pdb2pqr", "--ff=AMBER", "--with-ph=7.0", "--drop-water", "--keep-chain",
           f"--apbs-input={apbs_input}", temp_input_pdb, output_pqr]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"已生成PQR文件：{output_pqr}")
        print(f"已生成APBS配置文件：{apbs_input}")
        logging.info(f"已生成PQR文件：{output_pqr}")
        logging.info(f"已生成APBS配置文件：{apbs_input}")
    except subprocess.CalledProcessError as e:
        print(f"PDB2PQR错误：{e.stderr}")
        logging.error(f"PDB2PQR错误：{e.stderr}")
        raise
    finally:
        if os.path.exists(temp_input_pdb):
            os.unlink(temp_input_pdb)

def modify_apbs_input(apbs_input, dx_output, pqr_input):
    """修改APBS配置文件"""
    with open(apbs_input, 'r') as f:
        lines = f.readlines()
    base_name = os.path.splitext(os.path.basename(dx_output))[0]
    dx_dir = os.path.dirname(dx_output)
    with open(apbs_input, 'w') as f:
        for line in lines:
            if line.strip().startswith("read"):
                f.write(line)
                f.write(f"    mol pqr {pqr_input}\n")
                continue
            if line.strip().startswith("mol pqr"):
                continue
            if line.strip().startswith("write pot dx"):
                f.write(f"    write pot dx {dx_dir}/{base_name}\n")
            else:
                f.write(line)
    print(f"已修改APBS配置文件：{apbs_input}")
    logging.info(f"已修改APBS配置文件：{apbs_input}")

def run_apbs(apbs_input, dx_output):
    """运行APBS"""
    cmd = ["apbs", apbs_input]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"APBS计算完成，生成电势文件：{dx_output}")
        logging.info(f"APBS计算完成，生成电势文件：{dx_output}")
    except subprocess.CalledProcessError as e:
        print(f"APBS错误：{e.stderr}")
        logging.error(f"APBS错误：{e.stderr}")
        raise

def run_hbplus(input_pdb, output_hbplus):
    """运行HBPLUS分析氢键"""
    output_dir = os.path.dirname(output_hbplus)
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(input_pdb):
        raise FileNotFoundError(f"HBPLUS输入文件未找到：{input_pdb}")

    cmd = [
        "/Users/zhishou/Documents/Xulong/Software/biosoft/hbplus",
        "-h", "2.7", "-d", "4.5", "-a", "90",
        "-o", output_hbplus, input_pdb
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=output_dir)
        if not os.path.exists(output_hbplus):
            raise FileNotFoundError(f"HBPLUS未生成输出文件：{output_hbplus}")
        print(f"HBPLUS成功完成，生成氢键文件：{output_hbplus}")
        logging.info(f"HBPLUS成功完成，生成氢键文件：{output_hbplus}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        base_name = os.path.splitext(os.path.basename(input_pdb))[0]
        default_output = os.path.join(os.path.dirname(input_pdb), f"{base_name}.hb2")
        cmd_fallback = ["/Users/zhishou/Documents/Xulong/Software/biosoft/hbplus", input_pdb]
        try:
            result = subprocess.run(cmd_fallback, check=True, capture_output=True, text=True,
                                   cwd=os.path.dirname(input_pdb))
            if os.path.exists(default_output):
                shutil.move(default_output, output_hbplus)
                print(f"HBPLUS回退命令成功，将 {default_output} 移动到 {output_hbplus}")
                logging.info(f"HBPLUS回退命令成功，将 {default_output} 移动到 {output_hbplus}")
            else:
                raise FileNotFoundError(f"HBPLUS未生成 .hb2 文件")
        except subprocess.CalledProcessError as e2:
            print(f"HBPLUS回退命令失败：{e2.stderr}")
            logging.error(f"HBPLUS回退命令失败：{e2.stderr}")
            raise

def parse_hbplus_output(hbplus_file, atom_ids):
    """解析HBPLUS输出，提取氢键强度"""
    hbond_values = np.zeros(len(atom_ids))
    with open(hbplus_file, 'r') as f:
        for line in f:
            if not (line.startswith('A') or line.startswith('B')):
                continue
            parts = line.strip().split()
            if len(parts) < 12:
                continue
            try:
                chain_id = parts[0][0]
                res_name = parts[3]
                res_num = int(parts[0][1:5].strip('-'))
                atom_name = parts[2]
                role = 'DONOR' if atom_name in ['N', 'NH1', 'NH2', 'NZ', 'OG', 'OH'] else 'ACCEPTOR'
                d_a_dist = float(parts[8])
                h_a_dist = float(parts[11])
                dha_angle = float(parts[10])
                if h_a_dist < 3.0 and dha_angle > 90:
                    dist_factor = math.exp(-h_a_dist / 2.5)
                    angle_factor = math.cos(math.radians(180 - dha_angle))
                    strength = dist_factor * angle_factor
                    for i, atom_id in enumerate(atom_ids):
                        if (atom_id[0] == chain_id and
                            atom_id[1] == res_num and
                            atom_id[2] == res_name and
                            atom_id[3] == atom_name):
                            hbond_values[i] = strength if role == 'DONOR' else -strength
                            break
            except (ValueError, IndexError):
                continue
    max_abs = max(abs(hbond_values.max()), abs(hbond_values.min()))
    if max_abs > 0:
        hbond_values = hbond_values / max_abs * 0.971694  # 归一化到[-0.971694, 0.821083]
    print(f"解析到 {np.sum(hbond_values != 0)} 个氢键相关原子")
    logging.info(f"解析到 {np.sum(hbond_values != 0)} 个氢键相关原子")
    return hbond_values

def map_hbonds_to_vertices(vertices, atom_coords, hbond_values, radius=3.0):
    """将氢键值映射到表面顶点"""
    if len(atom_coords) != len(hbond_values):
        raise ValueError(
            f"atom_coords 长度 {len(atom_coords)} 与 hbond_values 长度 {len(hbond_values)} 不匹配")
    hbond_vertex = np.zeros(len(vertices))
    for i, vertex in enumerate(vertices):
        distances = np.linalg.norm(atom_coords - vertex, axis=1)
        mask = distances < radius
        if np.any(mask):
            weights = 1.0 / (distances[mask] + 1e-6)
            hbond_vertex[i] = np.sum(hbond_values[mask] * weights) / np.sum(weights)
    return hbond_vertex

def run_msms(input_xyzr, output_prefix):
    """运行MSMS"""
    cmd = ["msms", "-if", input_xyzr, "-of", output_prefix, "-probe_radius", "3", "-density", "1.0"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"MSMS计算完成，生成网格文件：{output_prefix}.vert, {output_prefix}.face")
        logging.info(f"MSMS计算完成，生成网格文件：{output_prefix}.vert, {output_prefix}.face")
    except subprocess.CalledProcessError as e:
        print(f"MSMS错误：{e.stderr}")
        logging.error(f"MSMS错误：{e.stderr}")
        raise

def convert_msms_to_ply(vert_file, face_file, output_ply):
    """将MSMS的.vert和.face转换为PLY"""
    with open(vert_file, 'r') as f:
        lines = f.readlines()
        vertices = []
        normals = []
        for line in lines[3:]:
            parts = line.strip().split()
            if len(parts) >= 6:
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                normals.append([float(parts[3]), float(parts[4]), float(parts[5])])
    vertices = np.array(vertices)
    normals = np.array(normals)

    with open(face_file, 'r') as f:
        lines = f.readlines()
        faces = []
        for line in lines[3:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                faces.append([int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1])
    faces = np.array(faces)

    cells = [("triangle", faces)]
    point_data = {"nx": normals[:, 0], "ny": normals[:, 1], "nz": normals[:, 2]}
    mesh = meshio.Mesh(points=vertices, cells=cells, point_data=point_data)
    meshio.write(output_ply, mesh, file_format="ply", binary=False)
    print(f"已生成初始PLY文件：{output_ply}")
    logging.info(f"已生成初始PLY文件：{output_ply}")

def read_dx_file(dx_file):
    """读取APBS的DX文件"""
    with open(dx_file, 'r') as f:
        lines = f.readlines()

    nx, ny, nz = 0, 0, 0
    origin = None
    delta = np.zeros((3, 3))
    delta_count = 0
    data = []
    in_data = False

    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith("object 1 class gridpositions counts"):
            nx, ny, nz = map(int, line.split()[-3:])
            continue
        if line.startswith("origin"):
            origin = np.array(list(map(float, line.split()[-3:])))
            continue
        if "delta" in line.lower() and len(line.split()) >= 4:
            delta[delta_count] = list(map(float, line.split()[-3:]))
            delta_count += 1
            continue
        if line.startswith("object 3 class array"):
            in_data = True
            continue
        if in_data and not line.startswith("attribute"):
            try:
                values = [float(x) for x in line.split()]
                data.extend(values)
            except ValueError:
                continue

    if nx == 0 or ny == 0 or nz == 0:
        raise ValueError("DX文件解析失败：未找到有效的网格尺寸")
    if origin is None:
        raise ValueError("DX文件解析失败：未找到原点")
    if delta_count != 3:
        raise ValueError(f"DX文件解析失败：仅找到 {delta_count} 个delta行")
    if not data:
        raise ValueError("DX文件解析失败：未找到有效的数据部分")

    data = np.array(data).reshape(nx, ny, nz)
    x = origin[0] + np.arange(nx) * delta[0, 0]
    y = origin[1] + np.arange(ny) * delta[1, 1]
    z = origin[2] + np.arange(nz) * delta[2, 2]
    return (x, y, z), data

def interpolate_potential(vertices, dx_file):
    """将电势插值到顶点"""
    (x, y, z), potential = read_dx_file(dx_file)
    interpolator = RegularGridInterpolator(
        (x, y, z), potential, method='linear', bounds_error=False, fill_value=0.0
    )
    charges = interpolator(vertices)
    return charges

def compute_interface_atoms(complex_pdb, protein_pdb, rna_pdb, sasa_threshold=0.1):
    """使用FreeSASA计算界面原子"""
    def create_temp_pdb(pdb_file):
        with open(pdb_file, 'r', encoding='utf-8') as f:
            content = f.read()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdb', mode='w', encoding='ascii').name
        with open(temp_file, 'w', encoding='ascii') as tf:
            tf.write(content)
        return temp_file

    try:
        complex_structure = freesasa.Structure(complex_pdb)
    except UnicodeEncodeError:
        complex_pdb_temp = create_temp_pdb(complex_pdb)
        complex_structure = freesasa.Structure(complex_pdb_temp)

    try:
        protein_structure = freesasa.Structure(protein_pdb)
    except UnicodeEncodeError:
        protein_pdb_temp = create_temp_pdb(protein_pdb)
        protein_structure = freesasa.Structure(protein_pdb_temp)

    try:
        rna_structure = freesasa.Structure(rna_pdb)
    except UnicodeEncodeError:
        rna_pdb_temp = create_temp_pdb(rna_pdb)
        rna_structure = freesasa.Structure(rna_pdb_temp)

    params = freesasa.Parameters({'probe-radius': 1.4})
    complex_result = freesasa.calc(complex_structure, params)
    protein_result = freesasa.calc(protein_structure, params)
    rna_result = freesasa.calc(rna_structure, params)

    for temp_file in [locals().get('complex_pdb_temp'), locals().get('protein_pdb_temp'), locals().get('rna_pdb_temp')]:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)

    interface_atoms = []
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", complex_pdb)

    atom_coords = []
    atom_ids = []
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname()
                for atom in residue:
                    atom_coords.append(atom.get_coord())
                    atom_ids.append((chain.id, residue.id[1], res_name, atom.get_name()))

    atom_coords = np.array(atom_coords)

    for i in range(complex_structure.nAtoms()):
        chain_id = complex_structure.chainLabel(i)
        atom_name = complex_structure.atomName(i)
        res_num = complex_structure.residueNumber(i)
        delta_sasa = 0.0

        if any(protein_structure.chainLabel(j) == chain_id for j in range(protein_structure.nAtoms())):
            for j in range(protein_structure.nAtoms()):
                if (protein_structure.chainLabel(j) == chain_id and
                    protein_structure.atomName(j) == atom_name and
                    protein_structure.residueNumber(j) == res_num):
                    delta_sasa = protein_result.atomArea(j) - complex_result.atomArea(i)
                    break
        elif any(rna_structure.chainLabel(j) == chain_id for j in range(rna_structure.nAtoms())):
            for j in range(rna_structure.nAtoms()):
                if (rna_structure.chainLabel(j) == chain_id and
                    rna_structure.atomName(j) == atom_name and
                    rna_structure.residueNumber(j) == res_num):
                    delta_sasa = rna_result.atomArea(j) - complex_result.atomArea(i)
                    break

        if delta_sasa > sasa_threshold:
            atom_id = atom_ids[i]
            res_name = atom_id[2]
            hphob = hydrophobicity_scale.get(res_name, 0.0)
            interface_atoms.append({
                "molecule": "protein" if is_aa(res_name, standard=True) else "rna",
                "chain": atom_id[0],
                "residue": res_name,
                "resid": atom_id[1],
                "atom": atom_id[3],
                "delta_sasa": delta_sasa,
                "hydrophobicity": hphob
            })

    iface_values = np.array([1.0 if x["delta_sasa"] > sasa_threshold else 0.0 for x in interface_atoms])
    print(f"识别到 {len(interface_atoms)} 个界面原子")
    logging.info(f"识别到 {len(interface_atoms)} 个界面原子")
    return atom_coords, iface_values, atom_ids, interface_atoms

def map_hydrophobicity_to_vertices(vertices, atom_coords, atom_ids, sigma=1.5, cutoff=5.0):
    """将亲疏水性映射到表面顶点"""
    hphob_vertex = np.zeros(len(vertices))
    atom_residues = [(atom_id[0], atom_id[1], atom_id[2]) for atom_id in atom_ids]

    for i, vertex in enumerate(vertices):
        distances = np.linalg.norm(atom_coords - vertex, axis=1)
        mask = distances < cutoff
        if np.any(mask):
            weights = np.exp(-distances[mask] ** 2 / (2 * sigma ** 2))
            hphob_values = np.array([hydrophobicity_scale.get(atom_residues[j][2], 0.0) for j in np.where(mask)[0]])
            hphob_vertex[i] = np.sum(hphob_values * weights) / np.sum(weights)
    return hphob_vertex

def map_iface_to_vertices(vertices, atom_coords, iface_values, atom_ids, interface_atoms, radius=5.0):
    """将界面标签映射到表面顶点"""
    iface_vertex = np.zeros(len(vertices))
    interface_indices = [i for i, atom in enumerate(interface_atoms) if atom["delta_sasa"] > 0]
    interface_coords = atom_coords[interface_indices]

    for i, vertex in enumerate(vertices):
        if len(interface_coords) == 0:
            continue
        distances = np.linalg.norm(interface_coords - vertex, axis=1)
        mask = distances < radius
        if np.any(mask):
            iface_vertex[i] = 1.0
    return iface_vertex

def main():
    parser = argparse.ArgumentParser(description="计算RRM-RNA复合物的界面、静电势、亲疏水性和氢键属性")
    parser.add_argument("--input_pdb", required=True, help="输入PDB文件路径")
    parser.add_argument("--output_dir", default="output", help="输出文件基础目录")
    parser.add_argument("--apbs_path", default="apbs", help="APBS可执行文件路径")
    parser.add_argument("--pdb2pqr_path", default="pdb2pqr", help="PDB2PQR可执行文件路径")
    parser.add_argument("--msms_path", default="msms", help="MSMS可执行文件路径")
    parser.add_argument("--pdb_to_xyzr_path", default="pdb_to_xyzr", help="pdb_to_xyzr可执行文件路径")
    parser.add_argument("--hbplus_path", default="hbplus", help="HBPLUS可执行文件路径")
    parser.add_argument("--sasa_threshold", type=float, default=0.1, help="界面原子识别的SASA阈值")
    args = parser.parse_args()

    if not os.path.exists(args.input_pdb):
        raise FileNotFoundError(f"未找到输入PDB文件：{args.input_pdb}")

    paths = get_file_paths(args.input_pdb, args.output_dir)

    # 检查依赖工具
    apbs_cmd = check_command("apbs", args.apbs_path)
    pdb2pqr_cmd = check_command("pdb2pqr", args.pdb2pqr_path)
    msms_cmd = check_command("msms", args.msms_path)
    pdb_to_xyzr_cmd = check_command("pdb_to_xyzr", args.pdb_to_xyzr_path)
    hbplus_cmd = check_command("hbplus", args.hbplus_path)

    print("步骤1：清理PDB文件...")
    clean_pdb_file(args.input_pdb, paths["clean_pdb"])

    print("步骤2：识别蛋白质和RNA链...")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", paths["clean_pdb"])
    protein_chains, rna_chains = get_protein_rna_chains(structure)
    if not protein_chains or not rna_chains:
        raise ValueError("未找到蛋白质或RNA链")

    print(f"找到 {len(protein_chains)} 个蛋白质链: {protein_chains}")
    print(f"找到 {len(rna_chains)} 个RNA链: {rna_chains}")
    logging.info(f"找到 {len(protein_chains)} 个蛋白质链: {protein_chains}")
    logging.info(f"找到 {len(rna_chains)} 个RNA链: {rna_chains}")

    print("步骤3：分离蛋白质和RNA链...")
    split_chains(paths["clean_pdb"], paths["protein_pdb"], paths["rna_pdb"], protein_chains, rna_chains)

    print("步骤4：运行PDB2PQR...")
    run_pdb2pqr(paths["clean_pdb"], paths["pqr"], paths["apbs_in"])
    run_pdb2pqr(paths["protein_pdb"], paths["protein_pqr"], paths["protein_apbs_in"])
    run_pdb2pqr(paths["rna_pdb"], paths["rna_pqr"], paths["rna_apbs_in"])

    print("步骤5：转换为PDB...")
    convert_pqr_to_pdb(paths["pqr"], paths["pqr_pdb"])
    convert_pqr_to_pdb(paths["protein_pqr"], paths["protein_pqr_pdb"])
    convert_pqr_to_pdb(paths["rna_pqr"], paths["rna_pqr_pdb"])

    print("步骤6：修改APBS配置文件...")
    modify_apbs_input(paths["apbs_in"], paths["dx"], paths["pqr_pdb"])

    print("步骤7：运行APBS...")
    run_apbs(paths["apbs_in"], paths["dx"])

    print("步骤8：运行HBPLUS...")
    run_hbplus(paths["pqr_pdb"], paths["hbplus_out"])

    print("步骤9：运行MSMS...")
    subprocess.run([pdb_to_xyzr_cmd, paths["pqr_pdb"]], stdout=open(paths["xyzr"], "w"), check=True)
    run_msms(paths["xyzr"], os.path.splitext(paths["vert"])[0])
    convert_msms_to_ply(paths["vert"], paths["face"], paths["ply"])

    print("步骤10：计算界面原子...")
    atom_coords, iface_values, atom_ids, interface_atoms = compute_interface_atoms(
        paths["pqr_pdb"], paths["protein_pqr_pdb"], paths["rna_pqr_pdb"], args.sasa_threshold
    )

    print("步骤11：解析HBPLUS输出...")
    hbond_values = parse_hbplus_output(paths["hbplus_out"], atom_ids)

    print("步骤12：保存界面原子信息...")
    if interface_atoms:
        df = pd.DataFrame(interface_atoms)
        df.to_csv(paths["interface_csv"], index=False)
        print(f"界面原子已保存到 {paths['interface_csv']}")
        logging.info(f"界面原子已保存到 {paths['interface_csv']}")
        print(f"共找到 {len(interface_atoms)} 个界面原子")
        logging.info(f"共找到 {len(interface_atoms)} 个界面原子")

    print("步骤13：整合数据到PLY文件...")
    mesh = meshio.read(paths["ply"])
    vertices = mesh.points
    normals = np.column_stack([mesh.point_data["nx"], mesh.point_data["ny"], mesh.point_data["nz"]])

    charges = interpolate_potential(vertices, paths["dx"])
    charges = np.nan_to_num(charges, nan=0.0, posinf=0.0, neginf=0.0)

    hphob = map_hydrophobicity_to_vertices(vertices, atom_coords, atom_ids)
    hphob = np.nan_to_num(hphob, nan=0.0, posinf=0.0, neginf=0.0)

    iface = map_iface_to_vertices(vertices, atom_coords, iface_values, atom_ids, interface_atoms)
    iface = np.nan_to_num(iface, nan=0.0, posinf=0.0, neginf=0.0)

    hbonds = map_hbonds_to_vertices(vertices, atom_coords, hbond_values)
    hbonds = np.nan_to_num(hbonds, nan=0.0, posinf=0.0, neginf=0.0)

    point_data = {
        "nx": normals[:, 0], "ny": normals[:, 1], "nz": normals[:, 2],
        "charge": charges, "hphob": hphob, "iface": iface, "hbond": hbonds
    }

    new_mesh = meshio.Mesh(points=vertices, cells=mesh.cells, point_data=point_data)
    meshio.write(paths["output_ply"], new_mesh, file_format="ply", binary=False)
    print(f"处理完成！已生成最终PLY文件：{paths['output_ply']}")
    logging.info(f"处理完成！已生成最终PLY文件：{paths['output_ply']}")
    print(f"hbond 值范围：{hbonds.min():.6f} 到 {hbonds.max():.6f}")
    logging.info(f"hbond 值范围：{hbonds.min():.6f} 到 {hbonds.max():.6f}")

if __name__ == "__main__":
    main()