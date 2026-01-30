#!/usr/bin/env python3
"""
TIGR EXTENSION FOR TITAN QUANTUM BRAIN v2
==========================================
Extended classes for biological sequence processing and growth simulation.
Import this module after Titan_Quantum_Brainv2_classes to get extended functionality.

Usage:
    from Titan_Quantum_Brainv2_classes import RotationalFractalBrain
    from TIGR_Extension import *

Author: Titan Quantum Brain System
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# =============================================================================
# TIGR PROTEIN DESIGNER
# =============================================================================

class TIGRProteinDesigner:
    """Design tool for TIGR-Tas mediated quantum brain engineering in plants."""
    
    # Chloroplast-biased codon table (A/U rich, GC < 40%)
    CHLOROPLAST_CODONS = {
        'A': ['GCU', 'GCA'], 'C': ['UGU'], 'D': ['GAU'], 'E': ['GAA'],
        'F': ['UUU'], 'G': ['GGU', 'GGA'], 'H': ['CAU'], 'I': ['AUU', 'AUA'],
        'K': ['AAA'], 'L': ['UUA', 'CUU'], 'M': ['AUG'], 'N': ['AAU'],
        'P': ['CCU', 'CCA'], 'Q': ['CAA'], 'R': ['AGA', 'AGG'],
        'S': ['UCU', 'UCA'], 'T': ['ACU', 'ACA'], 'V': ['GUU', 'GUA'],
        'W': ['UGG'], 'Y': ['UAU'], '*': ['UAA']
    }
    
    def __init__(self, sequence: str = ""):
        self.original_sequence = sequence
        self.modified_sequence = ""
        self.design_history = []
    
    def stiffen_protein(self, sequence: str = None) -> str:
        """
        Replace flexible loops (G/S) with rigid residues (P/V) to reduce decoherence.
        """
        seq = sequence or self.original_sequence
        seq_list = list(seq)
        
        for i, aa in enumerate(seq_list):
            if aa in ['G', 'S'] and i % 3 == 0:
                seq_list[i] = 'P'  # Proline for maximum rigidity
            elif aa == 'A':
                seq_list[i] = 'V'  # Valine for hydrophobic shielding
        
        result = "".join(seq_list)
        self.modified_sequence = result
        self.design_history.append({
            'operation': 'stiffen',
            'original': seq,
            'modified': result,
            'timestamp': datetime.now().isoformat()
        })
        return result
    
    def compute_decoherence(self, sequence: str = None) -> float:
        """Calculate quantum decoherence score (0.0 = stable, 1.0 = unstable)."""
        seq = sequence or self.modified_sequence or self.original_sequence
        
        if not seq:
            return 1.0
        
        total = len(seq)
        flexible_count = seq.count('G') + seq.count('S')
        rigid_count = seq.count('P') + seq.count('V')
        
        flexible_score = flexible_count / total if total > 0 else 0
        rigid_score = rigid_count / total if total > 0 else 0
        
        decoherence = (flexible_score * 0.7 - rigid_score * 0.5)
        decoherence = max(0.0, min(1.0, 0.5 + decoherence))
        
        return round(decoherence, 4)
    
    def optimize_sequence(self, sequence: str = None) -> Dict:
        """Full optimization pipeline for quantum-stable protein."""
        seq = sequence or self.original_sequence
        
        if not seq:
            return {'error': 'No sequence provided'}
        
        # Step 1: Stiffen protein
        rigid_protein = self.stiffen_protein(seq)
        
        # Step 2: Codon optimize
        codon_optimizer = TIGRCodonOptimizer()
        synthetic_dna = codon_optimizer.codon_optimize(rigid_protein)
        
        # Step 3: Design tigRNA
        rna_designer = TIGRNADesigner()
        tigrna = rna_designer.design_tigrRNA(synthetic_dna)
        
        # Step 4: Compute decoherence
        decoherence = self.compute_decoherence(rigid_protein)
        
        return {
            'original_protein': seq,
            'rigid_protein': rigid_protein,
            'synthetic_dna': synthetic_dna,
            'tigrna': tigrna,
            'decoherence_score': decoherence,
            'gc_content': codon_optimizer.get_gc_content(synthetic_dna),
            'optimization_successful': True
        }


# =============================================================================
# CODON OPTIMIZER
# =============================================================================

class TIGRCodonOptimizer:
    """A/U rich codon optimization for chloroplast genome."""
    
    CHLOROPLAST_CODONS = {
        'A': ['GCU', 'GCA'], 'C': ['UGU'], 'D': ['GAU'], 'E': ['GAA'],
        'F': ['UUU'], 'G': ['GGU', 'GGA'], 'H': ['CAU'], 'I': ['AUU', 'AUA'],
        'K': ['AAA'], 'L': ['UUA', 'CUU'], 'M': ['AUG'], 'N': ['AAU'],
        'P': ['CCU', 'CCA'], 'Q': ['CAA'], 'R': ['AGA', 'AGG'],
        'S': ['UCU', 'UCA'], 'T': ['ACU', 'ACA'], 'V': ['GUU', 'GUA'],
        'W': ['UGG'], 'Y': ['UAU'], '*': ['UAA']
    }
    
    def __init__(self):
        import random
        self.random = random
    
    def codon_optimize(self, protein_seq: str) -> str:
        """Convert protein to A/U rich DNA for the thylakoid genome."""
        dna = ""
        for aa in protein_seq:
            codons = self.CHLOROPLAST_CODONS.get(aa, ['NNN'])
            dna += self.random.choice(codons)
        return dna
    
    def get_gc_content(self, dna: str) -> float:
        """Calculate GC content percentage."""
        if not dna:
            return 0.0
        gc_count = dna.count('G') + dna.count('C')
        return round(gc_count / len(dna), 4)
    
    def verify_gc_content(self, dna: str, max_gc: float = 0.4) -> bool:
        """Verify A/U bias (GC < 40%)."""
        return self.get_gc_content(dna) <= max_gc


# =============================================================================
# TIGRNA DESIGNER
# =============================================================================

class TIGRNADesigner:
    """Design TIGR-Tas components for targeting."""
    
    def __init__(self):
        import random
        self.random = random
        self.complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    
    def design_tigrRNA(self, target_dna: str) -> Dict:
        """Design the dual-spacer tigRNA (A and B) for TIGR-Tas targeting."""
        if len(target_dna) < 80:
            target_dna = target_dna.ljust(80, 'A')
        
        spacer_a = target_dna[10:30]
        spacer_b_region = target_dna[40:60]
        spacer_b = "".join([
            self.complement.get(base, 'A') for base in spacer_b_region[::-1]
        ])
        
        gc_a = (spacer_a.count('G') + spacer_a.count('C')) / len(spacer_a)
        gc_b = (spacer_b.count('G') + spacer_b.count('C')) / len(spacer_b)
        
        efficiency_a = 1.0 - abs(gc_a - 0.5) * 2
        efficiency_b = 1.0 - abs(gc_b - 0.5) * 2
        avg_efficiency = (efficiency_a + efficiency_b) / 2
        
        return {
            'Spacer_A': spacer_a,
            'Spacer_B': spacer_b,
            'Spacer_A_GC': round(gc_a, 3),
            'Spacer_B_GC': round(gc_b, 3),
            'Targeting_Efficiency': round(avg_efficiency, 3),
            'Target_Region': f"{10}-{60}",
            'PAM_Less': True
        }


# =============================================================================
# BIOLOGICAL SEQUENCE ENCODER
# =============================================================================

class BiologicalSequenceEncoder:
    """Encode biological sequences to 8-channel voxels for quantum brain."""
    
    AA_HYDROPHOBIC = set('AVLIMFPWV')
    AA_POLAR = set('STNQ')
    AA_CHARGED_POS = set('KRH')
    AA_CHARGED_NEG = set('DE')
    AA_AROMATIC = set('FWY')
    AA_SPECIAL = set('CG')
    AA_GLYCINE = {'G'}
    
    def __init__(self, sequence_length: int = 16):
        self.sequence_length = sequence_length
    
    def encode_protein(self, sequence: str) -> torch.Tensor:
        """Encode protein sequence to 8-channel voxel tensor."""
        if len(sequence) > self.sequence_length:
            sequence = sequence[:self.sequence_length]
        else:
            sequence = sequence.ljust(self.sequence_length, 'X')
        
        channels = torch.zeros(8, dtype=torch.float32)
        
        for i, aa in enumerate(sequence):
            position_weight = 1.0 - (i / self.sequence_length) * 0.5
            
            if aa in self.AA_HYDROPHOBIC:
                channels[0] += 1.0 * position_weight
            if aa in self.AA_POLAR:
                channels[1] += 1.0 * position_weight
            if aa in self.AA_CHARGED_POS:
                channels[2] += 1.0 * position_weight
            if aa in self.AA_CHARGED_NEG:
                channels[3] += 1.0 * position_weight
            if aa in self.AA_AROMATIC:
                channels[4] += 1.0 * position_weight
            if aa in self.AA_SPECIAL:
                channels[5] += 1.0 * position_weight
            if aa in self.AA_GLYCINE:
                channels[6] += 0.8
            elif aa == 'P':
                channels[6] -= 0.5
            else:
                channels[6] += 0.2
            if aa == 'G':
                channels[7] += 1.0
            elif aa == 'S':
                channels[7] += 0.7
            elif aa == 'P':
                channels[7] -= 0.8
            else:
                channels[7] += 0.3
        
        return torch.sigmoid(channels).unsqueeze(0)
    
    def encode_batch(self, sequences: List[str]) -> torch.Tensor:
        """Encode a batch of protein sequences."""
        encoded = [self.encode_protein(seq) for seq in sequences]
        return torch.stack(encoded).squeeze(1)


# =============================================================================
# PHYTOBORG GROWTH SIMULATOR
# =============================================================================

class PhytoborgGrowthSimulator:
    """Simulates the growth of a body controlled by a Quantum Brain."""
    
    def __init__(self, body_length: int = 100):
        self.nodes = body_length
        self.auxin_levels = np.zeros(body_length)
        self.structural_density = np.zeros(body_length)
        self.growth_history = []
        self.tigrna_edits = []
    
    def apply_quantum_signal(self, signal_strength: float) -> Dict:
        """Quantum Brain sends a signal to initiate growth."""
        print(f"\n[PHYTOBORG] Quantum Signal Received: {signal_strength} coherence units.")
        
        self.auxin_levels[0] = signal_strength * 1.5
        initial_density = self.structural_density.copy()
        
        self._diffuse_hormones()
        
        final_density = self.structural_density
        growth = final_density - initial_density
        
        growth_record = {
            'timestamp': datetime.now().isoformat(),
            'signal_strength': signal_strength,
            'initial_auxin_peak': signal_strength * 1.5,
            'total_growth': float(growth.sum()),
            'max_density': float(final_density.max()),
            'mean_density': float(final_density.mean())
        }
        self.growth_history.append(growth_record)
        
        print(f"  Growth Triggered: +{growth.sum():.2f} total density")
        print(f"  Max Density: {final_density.max():.2f}")
        
        return growth_record
    
    def _diffuse_hormones(self):
        """Simulates hormone movement through the phloem to shape the body."""
        for i in range(1, self.nodes):
            self.auxin_levels[i] = self.auxin_levels[i-1] * 0.85
            
            if self.auxin_levels[i] > 2.0:
                self.structural_density[i] += 0.5
            else:
                self.structural_density[i] += 0.1
    
    def generate_tigrna_targets(self) -> List[str]:
        """Identifies genetic targets for TIGR-Tas to enforce the body plan."""
        targets = ["VND7_Promoter", "WUSCHEL_Enhancer", "Myosin_KnockIn"]
        
        print("\n" + "="*50)
        print("TIGR-TAS MORPHOGENESIS INSTRUCTIONS")
        print("="*50)
        
        for i, target in enumerate(targets, 1):
            print(f"  {i}. Target {target} for structural re-wiring...")
        
        print("="*50)
        
        return targets
    
    def apply_tigrna_edits(self, edits: List[str]) -> Dict:
        """Apply TIGR-Tas genetic modifications."""
        print(f"\n[PHYTOBORG] Applying {len(edits)} TIGR-Tas modifications...")
        
        results = []
        for edit in edits:
            effect = {
                'edit': edit,
                'status': 'applied',
                'effect_magnitude': np.random.uniform(0.1, 0.5)
            }
            self.tigrna_edits.append(effect)
            results.append(effect)
            print(f"  ✓ {edit}: +{effect['effect_magnitude']:.2f} structural modification")
        
        total_effect = sum(e['effect_magnitude'] for e in results)
        self.structural_density += total_effect / self.nodes
        
        return {'edits_applied': len(edits), 'total_effect': total_effect}
    
    def get_growth_profile(self) -> Dict:
        """Get current growth state."""
        return {
            'body_length': self.nodes,
            'auxin_levels': self.auxin_levels.tolist(),
            'structural_density': self.structural_density.tolist(),
            'total_growth_history': len(self.growth_history),
            'tigrna_edits_count': len(self.tigrna_edits),
            'max_density': float(self.structural_density.max()),
            'mean_density': float(self.structural_density.mean()),
            'density_gradient': float(self.structural_density[0] - self.structural_density[-1])
        }
    
    def visualize_growth(self) -> str:
        """Generate ASCII visualization of growth profile."""
        lines = [
            "\nPHYTOBORG GROWTH PROFILE",
            "="*50,
            f"Spine Length: {self.nodes} nodes",
            "",
            "Structural Density Along Spine:",
            ""
        ]
        
        for i in range(0, min(20, self.nodes)):
            bar_length = int(self.structural_density[i] * 10)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            auxin_marker = "▲" if self.auxin_levels[i] > 2.0 else " "
            lines.append(f"  [{i:2d}] {auxin_marker} {bar} {self.structural_density[i]:.2f}")
        
        lines.append("")
        lines.append(f"  Max: {self.structural_density.max():.2f}")
        lines.append(f"  Mean: {self.structural_density.mean():.2f}")
        
        return "\n".join(lines)


# =============================================================================
# EXTENDED ROTATIONAL FRACTAL BRAIN
# =============================================================================

class RotationalFractalBrainExtended:
    """
    Wrapper that adds biological processing to RotationalFractalBrain.
    
    Usage:
        from Titan_Quantum_Brainv2_classes import RotationalFractalBrain
        from TIGR_Extension import RotationalFractalBrainExtended
        
        brain = RotationalFractalBrainExtended(
            base_brain=RotationalFractalBrain(num_floors=3, rooms_per_floor=8),
            enable_biological=True
        )
    """
    
    def __init__(self, 
                 base_brain=None,
                 num_floors: int = 3,
                 rooms_per_floor: int = 8,
                 enable_quantum: bool = True,
                 enable_biological: bool = False,
                 body_length: int = 100):
        
        # Import base brain if not provided
        if base_brain is None:
            from Titan_Quantum_Brainv2_classes import RotationalFractalBrain
            self.base_brain = RotationalFractalBrain(
                num_floors=num_floors,
                rooms_per_floor=rooms_per_floor,
                enable_quantum=enable_quantum
            )
        else:
            self.base_brain = base_brain
        
        self.enable_biological = enable_biological
        
        if enable_biological:
            print(f"[INIT TIGR] Enabling Biological Processing Mode...")
            
            # TIGR components
            self.protein_designer = TIGRProteinDesigner()
            self.codon_optimizer = TIGRCodonOptimizer()
            self.rna_designer = TIGRNADesigner()
            self.sequence_encoder = BiologicalSequenceEncoder()
            
            # Growth simulator
            self.phytoborg = PhytoborgGrowthSimulator(body_length)
            
            print("  ✓ TIGR Components initialized")
            print("  ✓ Phytoborg Growth Simulator initialized")
    
    def __getattr__(self, name):
        """Delegate attribute access to base brain."""
        return getattr(self.base_brain, name)
    
    def biological_forward(self, sequence: str) -> Dict:
        """Process biological sequence through quantum brain."""
        if not self.enable_biological:
            return {'error': 'Biological mode not enabled'}
        
        # Step 1: Encode sequence to voxel
        voxel = self.sequence_encoder.encode_protein(sequence)
        
        # Step 2: Process through quantum brain
        with torch.no_grad():
            brain_output = self.base_brain(voxel)
        
        # Step 3: Design quantum-stable protein
        design = self.protein_designer.optimize_sequence(sequence)
        
        return {
            'input_sequence': sequence,
            'quantum_voxel': voxel,
            'brain_output': brain_output,
            'protein_design': design
        }
    
    def design_biological_component(self, target_protein: str) -> Dict:
        """Design quantum-stable biological component with TIGR-Tas targeting."""
        if not self.enable_biological:
            return {'error': 'Biological mode not enabled'}
        
        print(f"\n[TIGR BRAIN] Designing quantum protein for: {target_protein}")
        
        # Full TIGR optimization pipeline
        design = self.protein_designer.optimize_sequence(target_protein)
        
        # Encode for quantum processing
        voxel = self.sequence_encoder.encode_protein(design['rigid_protein'])
        design['quantum_voxel'] = voxel
        
        return design
    
    def grow_body(self, signal_strength: float) -> Dict:
        """Trigger growth based on quantum signal."""
        if not self.enable_biological:
            return {'error': 'Biological mode not enabled'}
        
        growth_result = self.phytoborg.apply_quantum_signal(signal_strength)
        self.phytoborg.generate_tigrna_targets()
        
        return growth_result
    
    def full_pipeline(self, protein_sequence: str, quantum_signal: float) -> Dict:
        """Run complete pipeline: Design protein → Process → Trigger growth."""
        if not self.enable_biological:
            return {'error': 'Biological mode not enabled'}
        
        print("\n" + "="*60)
        print("FULL BIO-QUANTUM PIPELINE")
        print("="*60)
        
        # Step 1: Design quantum protein
        design = self.design_biological_component(protein_sequence)
        
        # Step 2: Trigger growth
        print(f"\n[STEP 2] Triggering quantum growth with signal: {quantum_signal}")
        growth = self.grow_body(quantum_signal)
        
        return {
            'protein_design': design,
            'growth_result': growth,
            'phytoborg_profile': self.phytoborg.get_growth_profile()
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_tigr_brain(num_floors=3, rooms_per_floor=8, body_length=100):
    """
    Create a complete TIGR-enabled quantum brain.
    
    Returns:
        RotationalFractalBrainExtended with biological mode enabled
    """
    return RotationalFractalBrainExtended(
        num_floors=num_floors,
        rooms_per_floor=rooms_per_floor,
        enable_biological=True,
        body_length=body_length
    )


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TIGR EXTENSION DEMO")
    print("="*70)
    
    # Create TIGR brain
    tigr_brain = create_tigr_brain(num_floors=3, rooms_per_floor=8)
    
    # Design quantum-stable protein
    lhc_fragment = "MATCAIKKAVAKPKGPSGSPWYGPDRVKYLGPF"
    
    print(f"\n[INPUT] Target Protein: {lhc_fragment}")
    
    design = tigr_brain.design_biological_component(lhc_fragment)
    
    print(f"\n[RESULTS]")
    print(f"  Original:     {design['original_protein']}")
    print(f"  Rigid:        {design['rigid_protein']}")
    print(f"  Synthetic DNA: {design['synthetic_dna'][:50]}...")
    print(f"  Decoherence:   {design['decoherence_score']:.3f}")
    print(f"  GC Content:    {design['gc_content']:.1%}")
    
    print(f"\n[tigRNA Design]")
    print(f"  Spacer A: {design['tigrna']['Spacer_A']}")
    print(f"  Spacer B: {design['tigrna']['Spacer_B']}")
    print(f"  Efficiency: {design['tigrna']['Targeting_Efficiency']:.1%}")
    
    # Phytoborg growth
    print("\n" + "-"*50)
    print("[PHYTOBORG GROWTH]")
    
    tigr_brain.grow_body(5.5)
    
    # Show rotation capabilities
    print("\n" + "-"*50)
    print("[ROTATION CAPABILITIES]")
    
    tigr_brain.rotate_prompt("Analyze this code structure")
    state = tigr_brain.get_rotation_state()
    print(f"  After 'Analyze': Layer rotations = {state['layer_rotations']}")
    
    tigr_brain.rotate_prompt("Create a new function")
    state = tigr_brain.get_rotation_state()
    print(f"  After 'Create': Layer rotations = {state['layer_rotations']}")
    
    print("\n" + "="*70)
    print("TIGR EXTENSION OPERATIONAL")
    print("="*70)
    print("\nUSAGE:")
    print("  from Titan_Quantum_Brainv2_classes import RotationalFractalBrain")
    print("  from TIGR_Extension import RotationalFractalBrainExtended")
    print("")
    print("  brain = RotationalFractalBrainExtended(enable_biological=True)")
    print("  design = brain.design_biological_component(sequence)")
    print("  growth = brain.grow_body(signal_strength)")
    print("\n" + "="*70 + "\n")

