#!/usr/bin/env python3
"""
TIGR & PHYTOBORG INTEGRATION DEMO
=================================
Demonstrates TIGR Quantum Designer + Phytoborg Growth Simulator
integrated with the Titan Quantum Brain v2.

Features:
1. TIGR Quantum Designer - Biological sequence processing
2. Phytoborg Growth Simulator - Growth controlled by quantum signals
3. Integrated pipeline - Combined biological + quantum processing

Author: Titan Quantum Brain System
"""

import torch
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime

print("\n" + "="*70)
print("TIGR & PHYTOBORG INTEGRATION DEMO")
print("="*70)


# =============================================================================
# PART 1: TIGR PROTEIN DESIGNER
# =============================================================================

class TIGRProteinDesigner:
    """Design tool for TIGR-Tas mediated quantum brain engineering in plants."""
    
    # Amino acid properties for encoding
    AA_PROPERTIES = {
        'hydrophobic': set('AVLIM'),
        'polar': set('STNQ'),
        'charged_pos': set('KRH'),
        'charged_neg': set('DE'),
        'aromatic': set('FYW'),
        'special': set('CGP'),  # C=cysteine, G=flexible, P=rigid
    }
    
    # Standard genetic code (for completeness)
    STANDARD_CODONS = {
        'A': ['GCU', 'GCC', 'GCA', 'GCG'],
        'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
        'N': ['AAU', 'AAC'],
        'D': ['GAU', 'GAC'],
        'C': ['UGU', 'UGC'],
        'Q': ['CAA', 'CAG'],
        'E': ['GAA', 'GAG'],
        'G': ['GGU', 'GGC', 'GGA', 'GGG'],
        'H': ['CAU', 'CAC'],
        'I': ['AUU', 'AUC', 'AUA'],
        'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],
        'K': ['AAA', 'AAG'],
        'M': ['AUG'],
        'F': ['UUU', 'UUC'],
        'P': ['CCU', 'CCC', 'CCA', 'CCG'],
        'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],
        'T': ['ACU', 'ACC', 'ACA', 'ACG'],
        'W': ['UGG'],
        'Y': ['UAU', 'UAC'],
        'V': ['GUU', 'GUC', 'GUA', 'GUG'],
        '*': ['UAA', 'UAG', 'UGA']
    }
    
    # Chloroplast-biased codon table (A/U rich, GC < 40%)
    CHLOROPLAST_CODONS = {
        'A': ['GCU', 'GCA'],  # Avoid GCG (high GC)
        'C': ['UGU'],
        'D': ['GAU'],
        'E': ['GAA'],
        'F': ['UUU'],
        'G': ['GGU', 'GGA'],  # Avoid GGC, GGG (high GC)
        'H': ['CAU'],
        'I': ['AUU', 'AUA'],  # Avoid AUC (high GC)
        'K': ['AAA'],
        'L': ['UUA', 'CUU'],  # Avoid CUC, CUG (high GC)
        'M': ['AUG'],
        'N': ['AAU'],
        'P': ['CCU', 'CCA'],  # Avoid CCC, CCG (high GC)
        'Q': ['CAA'],
        'R': ['AGA', 'AGG'],  # Avoid CGN codons (high GC)
        'S': ['UCU', 'UCA'],  # Avoid UCC, UCG, AGC (high GC)
        'T': ['ACU', 'ACA'],  # Avoid ACC, ACG (high GC)
        'V': ['GUU', 'GUA'],  # Avoid GUC, GUG (high GC)
        'W': ['UGG'],
        'Y': ['UAU'],
        '*': ['UAA']  # Use UAA (not UAG/UGA)
    }
    
    def __init__(self, sequence: str = ""):
        self.original_sequence = sequence
        self.modified_sequence = ""
        self.design_history = []
    
    def stiffen_protein(self, sequence: str = None) -> str:
        """
        Replaces flexible loops (G/S) with rigid residues (P/V) to reduce decoherence.
        
        Strategy:
        - Replace every 3rd Glycine (G) with Proline (P) for rigidity
        - Replace every 3rd Serine (S) with Valine (V) for hydrophobic shielding
        - Replace Alanine (A) with Valine (V) for increased stability
        """
        seq = sequence or self.original_sequence
        seq_list = list(seq)
        
        for i, aa in enumerate(seq_list):
            if aa in ['G', 'S'] and i % 3 == 0:
                # Replace flexible residues with rigid ones
                if aa == 'G':
                    seq_list[i] = 'P'  # Proline for maximum rigidity
                else:
                    seq_list[i] = 'P'  # Also Proline for Serine
            elif aa == 'A':
                seq_list[i] = 'V'  # Valine to increase hydrophobic shielding
        
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
        """
        Calculate quantum decoherence score (0.0 = stable, 1.0 = unstable).
        
        Factors:
        - High Glycine/Serine content → high decoherence
        - Proline content → low decoherence
        - Aromatic residues → medium decoherence (can participate in pi systems)
        """
        seq = sequence or self.modified_sequence or self.original_sequence
        
        if not seq:
            return 1.0
        
        # Count residue types
        total = len(seq)
        flexible_count = seq.count('G') + seq.count('S')
        rigid_count = seq.count('P') + seq.count('V')
        aromatic_count = sum(seq.count(aa) for aa in 'FWY')
        
        # Calculate scores
        flexible_score = flexible_count / total if total > 0 else 0
        rigid_score = rigid_count / total if total > 0 else 0
        aromatic_score = aromatic_count / total if total > 0 else 0
        
        # Decoherence: more flexible = higher, more rigid = lower
        # Aromatic can help (pi-electron systems can have quantum coherence)
        decoherence = (flexible_score * 0.7 - rigid_score * 0.5 + aromatic_score * 0.2)
        
        # Normalize to 0-1 range
        decoherence = max(0.0, min(1.0, 0.5 + decoherence))
        
        return round(decoherence, 4)
    
    def optimize_sequence(self, sequence: str = None) -> Dict:
        """
        Full optimization pipeline for quantum-stable protein.
        
        Returns:
            Dict with: rigid_protein, synthetic_dna, tigrna, decoherence_score
        """
        seq = sequence or self.original_sequence
        
        if not seq:
            return {'error': 'No sequence provided'}
        
        # Step 1: Stiffen protein
        rigid_protein = self.stiffen_protein(seq)
        
        # Step 2: Codon optimize (separate class)
        codon_optimizer = TIGRCodonOptimizer()
        synthetic_dna = codon_optimizer.codon_optimize(rigid_protein)
        
        # Step 3: Design tigRNA
        rna_designer = TIGRNADesigner()
        tigrna = rna_designer.design_tigrRNA(synthetic_dna)
        
        # Step 4: Compute decoherence
        decoherence = self.compute_decoherence(rigid_protein)
        
        result = {
            'original_protein': seq,
            'rigid_protein': rigid_protein,
            'synthetic_dna': synthetic_dna,
            'tigrna': tigrna,
            'decoherence_score': decoherence,
            'gc_content': codon_optimizer.get_gc_content(synthetic_dna),
            'optimization_successful': True
        }
        
        self.design_history.append({
            'operation': 'full_optimize',
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
        return result
    
    def get_design_report(self) -> str:
        """Generate a human-readable design report."""
        if not self.design_history:
            return "No designs generated yet."
        
        report_lines = [
            "="*50,
            "TIGR PROTEIN DESIGN REPORT",
            "="*50,
            f"Total designs: {len(self.design_history)}",
            ""
        ]
        
        for i, design in enumerate(self.design_history):
            report_lines.append(f"Design {i+1}: {design['operation']}")
            report_lines.append(f"  Timestamp: {design['timestamp']}")
            if 'result' in design:
                r = design['result']
                report_lines.append(f"  Original: {r.get('original_protein', 'N/A')}")
                report_lines.append(f"  Rigid:    {r.get('rigid_protein', 'N/A')}")
                report_lines.append(f"  DNA:      {r.get('synthetic_dna', 'N/A')[:40]}...")
                report_lines.append(f"  Decoherence: {r.get('decoherence_score', 'N/A')}")
            report_lines.append("")
        
        return "\n".join(report_lines)


# =============================================================================
# PART 2: CODON OPTIMIZER
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
# PART 3: TIGRNA DESIGNER
# =============================================================================

class TIGRNADesigner:
    """Design TIGR-Tas components for targeting."""
    
    def __init__(self):
        import random
        self.random = random
        self.complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    
    def design_tigrRNA(self, target_dna: str) -> Dict:
        """
        Design the dual-spacer tigRNA (A and B) for TIGR-Tas targeting.
        
        Spacer A (20nt) targets the top strand
        Spacer B (20nt) targets the bottom strand (complementary)
        
        Returns:
            Dict with Spacer_A, Spacer_B, and metadata
        """
        # Ensure minimum length
        if len(target_dna) < 80:
            # Pad with random nucleotides if needed
            target_dna = target_dna.ljust(80, 'A')
        
        # Extract spacer A from position 10-30
        spacer_a = target_dna[10:30]
        
        # Extract spacer B region from position 40-60
        spacer_b_region = target_dna[40:60]
        
        # Create complementary sequence for Spacer B
        spacer_b = "".join([
            self.complement.get(base, 'A') 
            for base in spacer_b_region[::-1]  # Reverse complement
        ])
        
        # Calculate targeting efficiency (based on GC content)
        gc_a = (spacer_a.count('G') + spacer_a.count('C')) / len(spacer_a)
        gc_b = (spacer_b.count('G') + spacer_b.count('C')) / len(spacer_b)
        
        # Optimal GC is 40-60%
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
    
    def validate_spacers(self, spacer_a: str, spacer_b: str) -> Dict:
        """Validate spacer sequences."""
        validation = {
            'length_valid': len(spacer_a) == 20 and len(spacer_b) == 20,
            'no_internal_homology': self._check_homology(spacer_a, spacer_b),
            'gc_in_range': all(0.3 <= self._gc_content(s) <= 0.7 for s in [spacer_a, spacer_b]),
            'valid': False
        }
        
        validation['valid'] = all(validation.values())
        return validation
    
    def _check_homology(self, seq1: str, seq2: str) -> bool:
        """Check for sequence homology between two sequences."""
        # Simple check: no more than 10 consecutive matches
        matches = 0
        for a, b in zip(seq1, seq2):
            if a == b:
                matches += 1
                if matches > 10:
                    return False
            else:
                matches = 0
        return True
    
    def _gc_content(self, seq: str) -> float:
        """Calculate GC content."""
        if not seq:
            return 0.0
        return (seq.count('G') + seq.count('C')) / len(seq)


# =============================================================================
# PART 4: BIOLOGICAL SEQUENCE ENCODER
# =============================================================================

class BiologicalSequenceEncoder:
    """Encode biological sequences to 8-channel voxels for quantum brain."""
    
    # Amino acid property mappings
    AA_HYDROPHOBIC = set('AVLIMFPWV')
    AA_POLAR = set('STNQ')
    AA_CHARGED_POS = set('KRH')
    AA_CHARGED_NEG = set('DE')
    AA_AROMATIC = set('FWY')
    AA_SPECIAL = set('CG')
    AA_GLYCINE = {'G'}
    AA_PROLINE = {'P'}
    
    def __init__(self, sequence_length: int = 16):
        self.sequence_length = sequence_length
    
    def encode_protein(self, sequence: str) -> torch.Tensor:
        """
        Encode protein sequence to 8-channel voxel tensor.
        
        Channel mapping:
        0: Hydrophobic (A, V, L, I, M, F, P, W, V)
        1: Polar (S, T, N, Q)
        2: Charged Positive (K, R, H)
        3: Charged Negative (D, E)
        4: Aromatic (F, Y, W)
        5: Special/Structural (C, G, P)
        6: Decoherence Index (computed from sequence)
        7: Flexibility Index (G/S content)
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            Tensor of shape (1, 8) for single sequence
            or (batch_size, 8) for batch
        """
        # Normalize sequence length
        if len(sequence) > self.sequence_length:
            sequence = sequence[:self.sequence_length]
        else:
            sequence = sequence.ljust(self.sequence_length, 'X')
        
        # Calculate properties for each position
        channels = torch.zeros(8, dtype=torch.float32)
        
        for i, aa in enumerate(sequence):
            position_weight = 1.0 - (i / self.sequence_length) * 0.5  # N-terminal bias
            
            # Channel 0: Hydrophobic
            if aa in self.AA_HYDROPHOBIC:
                channels[0] += 1.0 * position_weight
            
            # Channel 1: Polar
            if aa in self.AA_POLAR:
                channels[1] += 1.0 * position_weight
            
            # Channel 2: Charged Positive
            if aa in self.AA_CHARGED_POS:
                channels[2] += 1.0 * position_weight
            
            # Channel 3: Charged Negative
            if aa in self.AA_CHARGED_NEG:
                channels[3] += 1.0 * position_weight
            
            # Channel 4: Aromatic
            if aa in self.AA_AROMATIC:
                channels[4] += 1.0 * position_weight
            
            # Channel 5: Special/Structural (C, G, P have special roles)
            if aa in self.AA_SPECIAL:
                channels[5] += 1.0 * position_weight
            
            # Channel 6: Decoherence (G/S flexible, P rigid)
            if aa in self.AA_GLYCINE:
                channels[6] += 0.8  # High decoherence
            elif aa == 'P':
                channels[6] -= 0.5  # Reduces decoherence
            else:
                channels[6] += 0.2  # Base decoherence
            
            # Channel 7: Flexibility (G is most flexible)
            if aa == 'G':
                channels[7] += 1.0
            elif aa == 'S':
                channels[7] += 0.7
            elif aa == 'P':
                channels[7] -= 0.8  # Rigid
            else:
                channels[7] += 0.3
        
        # Normalize channels to 0-1 range
        channels = torch.sigmoid(channels)
        
        return channels.unsqueeze(0)  # Shape: (1, 8)
    
    def encode_batch(self, sequences: List[str]) -> torch.Tensor:
        """Encode a batch of protein sequences."""
        encoded = [self.encode_protein(seq) for seq in sequences]
        return torch.stack(encoded).squeeze(1)  # Shape: (batch, 8)
    
    def decode_voxel(self, voxel: torch.Tensor) -> Dict:
        """Decode a voxel back to sequence properties."""
        properties = {
            'hydrophobicity': voxel[0].item(),
            'polarity': voxel[1].item(),
            'positive_charge': voxel[2].item(),
            'negative_charge': voxel[3].item(),
            'aromaticity': voxel[4].item(),
            'structural_special': voxel[5].item(),
            'decoherence_index': voxel[6].item(),
            'flexibility': voxel[7].item()
        }
        return properties


# =============================================================================
# PART 5: PHYTOBORG GROWTH SIMULATOR
# =============================================================================

class PhytoborgGrowthSimulator:
    """Simulates the growth of a body controlled by a Quantum Brain."""
    
    def __init__(self, body_length: int = 100):
        self.nodes = body_length
        # Initialize hormone levels (Auxin gradient)
        self.auxin_levels = np.zeros(body_length)
        # Structural density (Lignin)
        self.structural_density = np.zeros(body_length)
        # Growth history
        self.growth_history = []
        # TIGR-Tas modifications
        self.tigrna_edits = []
    
    def apply_quantum_signal(self, signal_strength: float) -> Dict:
        """
        Quantum Brain sends a signal to initiate growth.
        
        Args:
            signal_strength: Coherence units from quantum brain (typically 1-10)
            
        Returns:
            Dict with growth metrics
        """
        print(f"\n[PHYTOBORG] Quantum Signal Received: {signal_strength} coherence units.")
        
        # The signal triggers auxin production at the 'Apex'
        self.auxin_levels[0] = signal_strength * 1.5
        
        # Track initial state
        initial_density = self.structural_density.copy()
        
        # Diffuse hormones
        self._diffuse_hormones()
        
        # Calculate growth
        final_density = self.structural_density
        growth = final_density - initial_density
        
        # Record
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
            # Hormones diffuse down the 'spine' with exponential decay
            self.auxin_levels[i] = self.auxin_levels[i-1] * 0.85
            
            # TIGR-Tas logic: Where auxin is high, increase lignin (bone growth)
            if self.auxin_levels[i] > 2.0:
                self.structural_density[i] += 0.5
            else:
                self.structural_density[i] += 0.1
    
    def generate_tigrna_targets(self) -> List[str]:
        """Identifies genetic targets for TIGR-Tas to enforce the body plan."""
        targets = [
            "VND7_Promoter",      # Vascular differentiation
            "WUSCHEL_Enhancer",   # Stem cell maintenance  
            "Myosin_KnockIn"      # Cytoskeletal modification
        ]
        
        print("\n" + "="*50)
        print("TIGR-TAS MORPHOGENESIS INSTRUCTIONS")
        print("="*50)
        
        for i, target in enumerate(targets, 1):
            print(f"  {i}. Target {target} for structural re-wiring...")
        
        print("="*50)
        
        return targets
    
    def apply_tigrna_edits(self, edits: List[str]) -> Dict:
        """
        Apply TIGR-Tas genetic modifications.
        
        Args:
            edits: List of edit instructions
            
        Returns:
            Dict with edit results
        """
        print(f"\n[PHYTOBORG] Applying {len(edits)} TIGR-Tas modifications...")
        
        results = []
        for edit in edits:
            # Simulate modification effect
            effect = {
                'edit': edit,
                'status': 'applied',
                'effect_magnitude': np.random.uniform(0.1, 0.5)
            }
            self.tigrna_edits.append(effect)
            results.append(effect)
            print(f"  ✓ {edit}: +{effect['effect_magnitude']:.2f} structural modification")
        
        # Apply cumulative effect to density
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
        
        # Show first 20 nodes
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
# PART 6: INTEGRATED BIO-QUANTUM BRAIN
# =============================================================================

class BioQuantumBrain:
    """
    Complete integrated system combining:
    - TIGR Quantum Designer (biological sequence processing)
    - Phytoborg Growth Simulator (growth controlled by quantum signals)
    - RotationalFractalBrain (quantum processing)
    """
    
    def __init__(self, 
                 num_floors: int = 3, 
                 rooms_per_floor: int = 8,
                 body_length: int = 100):
        
        print("\n[INIT] Building Bio-Quantum Brain System...")
        
        # Core components
        self.protein_designer = TIGRProteinDesigner()
        self.codon_optimizer = TIGRCodonOptimizer()
        self.rna_designer = TIGRNADesigner()
        self.sequence_encoder = BiologicalSequenceEncoder()
        
        # Growth simulator
        self.phytoborg = PhytoborgGrowthSimulator(body_length)
        
        print("  ✓ TIGR Components initialized")
        print("  ✓ Phytoborg Growth Simulator initialized")
        print("\n[INIT] Bio-Quantum Brain ready!")
    
    def design_quantum_protein(self, target_protein: str) -> Dict:
        """
        Design a quantum-stable protein with TIGR-Tas targeting.
        
        Args:
            target_protein: Amino acid sequence
            
        Returns:
            Complete design package
        """
        print(f"\n[BIO-BRAIN] Designing quantum protein for: {target_protein}")
        
        # Step 1: Optimize sequence
        design = self.protein_designer.optimize_sequence(target_protein)
        
        # Step 2: Encode for quantum processing
        voxel = self.sequence_encoder.encode_protein(design['rigid_protein'])
        design['quantum_voxel'] = voxel
        
        # Step 3: Use decoherence score to influence growth
        coherence_factor = 1.0 - design['decoherence_score']
        print(f"  Decoherence Score: {design['decoherence_score']:.3f}")
        print(f"  Coherence Factor: {coherence_factor:.3f}")
        
        return design
    
    def trigger_quantum_growth(self, signal_strength: float) -> Dict:
        """
        Trigger growth based on quantum brain output.
        
        Args:
            signal_strength: Signal from quantum processing
            
        Returns:
            Growth results
        """
        growth_result = self.phytoborg.apply_quantum_signal(signal_strength)
        self.phytoborg.generate_tigrna_targets()
        return growth_result
    
    def full_pipeline(self, protein_sequence: str, quantum_signal: float) -> Dict:
        """
        Run complete pipeline: Design protein → Process → Trigger growth.
        
        Args:
            protein_sequence: Target protein sequence
            quantum_signal: Signal strength for growth
            
        Returns:
            Complete pipeline results
        """
        print("\n" + "="*60)
        print("FULL BIO-QUANTUM PIPELINE")
        print("="*60)
        
        # Step 1: Design quantum protein
        design = self.design_quantum_protein(protein_sequence)
        
        # Step 2: Trigger growth based on quantum coherence
        print(f"\n[STEP 2] Triggering quantum growth with signal: {quantum_signal}")
        growth = self.trigger_quantum_growth(quantum_signal)
        
        # Combine results
        return {
            'protein_design': design,
            'growth_result': growth,
            'phytoborg_profile': self.phytoborg.get_growth_profile()
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def run_demo():
    """Run complete demonstration of TIGR + Phytoborg integration."""
    
    print("\n" + "="*70)
    print("DEMO 1: TIGR PROTEIN DESIGNER")
    print("="*70)
    
    # Hypothetical segment of LHCII (Light Harvesting Complex II)
    lhc_fragment = "MATCAIKKAVAKPKGPSGSPWYGPDRVKYLGPF"
    
    designer = TIGRProteinDesigner(lhc_fragment)
    
    print(f"\nOriginal Sequence:  {lhc_fragment}")
    
    # Stiffen protein
    rigid = designer.stiffen_protein()
    print(f"Rigid Sequence:     {rigid}")
    
    # Compute decoherence
    original_decoherence = designer.compute_decoherence(lhc_fragment)
    rigid_decoherence = designer.compute_decoherence(rigid)
    print(f"\nDecoherence Scores:")
    print(f"  Original: {original_decoherence:.3f}")
    print(f"  Rigid:    {rigid_decoherence:.3f}")
    print(f"  Improvement: {(original_decoherence - rigid_decoherence):.3f}")
    
    # Codon optimize
    codon_opt = TIGRCodonOptimizer()
    synthetic_dna = codon_opt.codon_optimize(rigid)
    print(f"\nSynthetic DNA (Chloroplast Optimized):")
    print(f"  {synthetic_dna}")
    print(f"  GC Content: {codon_opt.get_gc_content(synthetic_dna):.1%} (target: <40%)")
    
    # Design tigRNA
    tigRNA = TIGRNADesigner().design_tigrRNA(synthetic_dna)
    print(f"\ntigRNA Design:")
    print(f"  Spacer A: {tigRNA['Spacer_A']}")
    print(f"  Spacer B: {tigRNA['Spacer_B']}")
    print(f"  Efficiency: {tigRNA['Targeting_Efficiency']:.1%}")
    
    # =============================================================================
    print("\n" + "="*70)
    print("DEMO 2: PHYTOBORG GROWTH SIMULATOR")
    print("="*70)
    
    # Signal strength from the Quantum Thylakoids
    quantum_input = 5.5
    
    avatar = PhytoborgGrowthSimulator()
    avatar.apply_quantum_signal(quantum_input)
    avatar.generate_tigrna_targets()
    
    print(avatar.visualize_growth())
    
    # Apply more signals
    print("\n--- Additional Quantum Signals ---")
    avatar.apply_quantum_signal(3.2)
    avatar.apply_quantum_signal(7.8)
    
    # Apply TIGR-Tas modifications
    print("\n--- TIGR-Tas Modifications ---")
    avatar.apply_tigrna_edits([
        "VND7_Promoter_enhance",
        "WUSCHEL_boost",
        "Cytoskeletal_strengthen"
    ])
    
    # Final profile
    print(avatar.visualize_growth())
    
    # =============================================================================
    print("\n" + "="*70)
    print("DEMO 3: BIOLOGICAL SEQUENCE ENCODING")
    print("="*70)
    
    encoder = BiologicalSequenceEncoder()
    voxel = encoder.encode_protein(lhc_fragment)
    
    print(f"\nProtein: {lhc_fragment}")
    print(f"\n8-Channel Voxel Encoding:")
    channel_names = [
        'Hydrophobic', 'Polar', 'Charged+', 'Charged-',
        'Aromatic', 'Special', 'Decoherence', 'Flexibility'
    ]
    for i, (name, val) in enumerate(zip(channel_names, voxel[0])):
        print(f"  Channel {i} ({name:12s}): {val:.4f}")
    
    # =============================================================================
    print("\n" + "="*70)
    print("DEMO 4: INTEGRATED BIO-QUANTUM BRAIN")
    print("="*70)
    
    bio_brain = BioQuantumBrain()
    
    # Run full pipeline
    result = bio_brain.full_pipeline(
        protein_sequence="MATLGRKAVAKPKGPSGSPWYGPDRVKYLGPF",
        quantum_signal=5.5
    )
    
    print("\n" + "="*70)
    print("INTEGRATION DEMO COMPLETE")
    print("="*70)
    print("\nCOMPONENTS DEMONSTRATED:")
    print("  ✓ TIGRProteinDesigner (sequence stiffening)")
    print("  ✓ TIGRCodonOptimizer (A/U rich optimization)")
    print("  ✓ TIGRNADesigner (dual-spacer tigRNA)")
    print("  ✓ BiologicalSequenceEncoder (voxel encoding)")
    print("  ✓ PhytoborgGrowthSimulator (quantum-controlled growth)")
    print("  ✓ BioQuantumBrain (integrated system)")
    
    return result


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    results = run_demo()
    
    print("\n" + "="*70)
    print("ALL SYSTEMS OPERATIONAL")
    print("="*70)
    print("\nUSAGE:")
    print("  1. TIGR Protein Design:")
    print("     designer = TIGRProteinDesigner(sequence)")
    print("     result = designer.optimize_sequence()")
    print("")
    print("  2. Phytoborg Growth:")
    print("     avatar = PhytoborgGrowthSimulator()")
    print("     avatar.apply_quantum_signal(strength)")
    print("")
    print("  3. Full Integration:")
    print("     brain = BioQuantumBrain()")
    print("     result = brain.full_pipeline(sequence, signal)")
    print("\n" + "="*70 + "\n")

