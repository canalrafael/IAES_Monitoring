#!/usr/bin/env python3
"""
clean_to_csv.py — Extrai dados PMU e DET dos arquivos data_detector*.txt
                   e gera CSVs limpos e organizados.

Saída:
  csv/data_detectorX_pmu.csv   — dados PMU limpos
  csv/data_detectorX_det.csv   — predições do detector
"""

import sys
from pathlib import Path


def ts_to_ms(ts_str):
    """Converte HH:MM:SS:mmm para milissegundos desde boot."""
    parts = ts_str.split(':')
    if len(parts) != 4:
        return None
    try:
        hh, mm, ss, ms = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        return hh * 3600000 + mm * 60000 + ss * 1000 + ms
    except ValueError:
        return None


def process_file(filepath, out_dir):
    """Processa um arquivo .txt e gera CSVs limpos."""
    fname = Path(filepath).stem  # ex: data_detector0

    pmu_out = out_dir / f"{fname}_pmu.csv"
    det_out = out_dir / f"{fname}_det.csv"

    in_pmu = False
    in_det = False
    pmu_header_seen = False
    det_header_seen = False

    pmu_count = 0
    det_count = 0
    skip_count = 0

    with open(filepath, 'r', errors='replace') as fin, \
         open(pmu_out, 'w') as fpmu, \
         open(det_out, 'w') as fdet:

        # Escreve headers
        fpmu.write("timestamp,cpu_cycles,instructions,cache_misses,branch_misses,l2_cache_access,label\n")
        fdet.write("status,probability\n")

        for line in fin:
            line = line.strip()

            if line == 'PMU_START':
                in_pmu = True
                pmu_header_seen = False
                continue
            elif line == 'PMU_END':
                in_pmu = False
                continue
            elif line == 'DET_START':
                in_det = True
                det_header_seen = False
                continue
            elif line == 'DET_END':
                in_det = False
                continue

            if in_pmu:
                if not pmu_header_seen:
                    pmu_header_seen = True
                    continue
                parts = line.split(',')
                if len(parts) != 8:
                    skip_count += 1
                    continue
                try:
                    core_id = int(parts[0])
                    timestamp = parts[1]
                    cpu_cycles = int(parts[2])
                    instructions = int(parts[3])
                    cache_misses = int(parts[4])
                    branch_misses = int(parts[5])
                    l2_cache = int(parts[6])
                    label = int(parts[7])

                    fpmu.write(f"{timestamp},{cpu_cycles},{instructions},{cache_misses},{branch_misses},{l2_cache},{label}\n")
                    pmu_count += 1
                except ValueError:
                    skip_count += 1

            if in_det:
                if not det_header_seen:
                    det_header_seen = True
                    continue
                parts = line.split(',')
                if len(parts) != 3:
                    skip_count += 1
                    continue
                try:
                    sample_idx = int(parts[0])
                    status = parts[1]
                    prob = int(parts[2])

                    fdet.write(f"{status},{prob}\n")
                    det_count += 1
                except ValueError:
                    skip_count += 1

    return pmu_count, det_count, skip_count, pmu_out, det_out


def main():
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "csv"
    out_dir.mkdir(exist_ok=True)

    files = sorted(script_dir.glob('data_detector*.txt'))
    if not files:
        print(f"⚠ Nenhum arquivo data_detector*.txt encontrado em {script_dir}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  Limpeza e conversão para CSV")
    print(f"  Encontrados {len(files)} arquivo(s)")
    print(f"  Saída: {out_dir}/")
    print(f"{'='*60}")

    total_pmu = 0
    total_det = 0

    for filepath in files:
        fname = filepath.name
        print(f"\n  📄 {fname}")

        pmu_n, det_n, skip_n, pmu_path, det_path = process_file(filepath, out_dir)
        total_pmu += pmu_n
        total_det += det_n

        print(f"    → {pmu_path.name}: {pmu_n:>8,} linhas PMU")
        print(f"    → {det_path.name}: {det_n:>8,} linhas DET")
        if skip_n > 0:
            print(f"    ⚠ {skip_n} linhas descartadas (boot/parse errors)")

    print(f"\n{'='*60}")
    print(f"  ✅ Concluído!")
    print(f"  📊 Total PMU: {total_pmu:,} linhas")
    print(f"  🤖 Total DET: {total_det:,} linhas")
    print(f"  📁 CSVs em:   {out_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
