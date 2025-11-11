"""
Mathlete Token (MLT) Implementation

Represents a research claim under verification in the Mathlete Chain.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import hashlib


@dataclass
class ValidationEvent:
    """Represents a single validation/replication attempt"""
    validator_id: str
    epoch: int
    success: bool
    proof_hash: str  # Hash of replication proof
    delta_tau: float  # Change in trust from this validation
    entropy_contribution: float  # Entropy added (if failure)


@dataclass
class MathleteToken:
    """
    Mathlete Token (MLT) - represents a research claim under verification.
    """
    claim_hash: str  # IPFS/Arweave hash of the research artifact
    uncertainty_sigma: float  # Standard deviation of model stability
    trust_tau: float  # Verified trust coefficient [0,1]
    gamma_t: float  # Entropy rate (concept drift)
    pod_id: str  # Field of study or domain
    minted_at: int  # Epoch timestamp
    last_updated: int  # Last validation epoch
    
    # Fields with defaults
    validation_events: List[ValidationEvent] = field(default_factory=list)
    
    # Temporal compounding metrics
    F_t: float = 0.0  # Current foresight efficiency
    L_t: float = 1.0  # Current trust leverage
    U_n: float = 1.0  # Compounded understanding
    
    # Parameters
    L_0: float = 1.0  # Baseline leverage
    k: float = 2.0  # Sensitivity constant
    tau_eq: float = 0.5  # Equilibrium trust threshold
    lambda_val: float = 0.1  # Learning rate
    
    def __post_init__(self):
        """Initialize token state"""
        if self.last_updated == 0:
            self.last_updated = self.minted_at
    
    def calculate_initial_value(self, k: float = 1.0) -> float:
        """
        Calculate initial informational value.
        
        P_0 = k * (τ_0 / σ_0)
        """
        if self.uncertainty_sigma == 0:
            return float('inf')
        return k * (self.trust_tau / self.uncertainty_sigma)
    
    def update_trust(self, delta_tau: float) -> None:
        """
        Update trust coefficient after validation.
        
        Args:
            delta_tau: Change in trust (positive for success, negative for failure)
        """
        self.trust_tau = max(0.0, min(1.0, self.trust_tau + delta_tau))
    
    def update_uncertainty(self, delta_sigma: float) -> None:
        """
        Update uncertainty after validation.
        
        Args:
            delta_sigma: Change in uncertainty (negative for success, positive for failure)
        """
        self.uncertainty_sigma = max(0.0, self.uncertainty_sigma + delta_sigma)
    
    def update_entropy(self, delta_gamma: float) -> None:
        """
        Update entropy rate.
        
        Args:
            delta_gamma: Change in entropy (positive for failures/inconsistencies)
        """
        self.gamma_t = max(0.0, self.gamma_t + delta_gamma)
    
    def calculate_trust_leverage(self) -> float:
        """
        Calculate trust leverage.
        
        L_t = L_0(1 + k(τ_t - τ_eq))
        """
        return self.L_0 * (1 + self.k * (self.trust_tau - self.tau_eq))
    
    def calculate_foresight_efficiency(self, V_t: float, T_t: float, C_t: float) -> float:
        """
        Calculate foresight efficiency.
        
        F_t = V_t / (T_t * C_t)
        """
        if T_t == 0 or C_t == 0:
            return 0.0
        return V_t / (T_t * C_t)
    
    def add_validation(self, validator_id: str, epoch: int, success: bool, 
                      proof_hash: str, delta_tau: float = 0.1,
                      entropy_contribution: float = 0.0) -> ValidationEvent:
        """
        Add a validation event and update token state.
        
        Args:
            validator_id: ID of validator
            epoch: Current epoch
            success: Whether replication succeeded
            proof_hash: Hash of replication proof
            delta_tau: Trust change (positive for success)
            entropy_contribution: Entropy added (for failures)
        
        Returns:
            ValidationEvent object
        """
        event = ValidationEvent(
            validator_id=validator_id,
            epoch=epoch,
            success=success,
            proof_hash=proof_hash,
            delta_tau=delta_tau if success else -delta_tau * 0.5,
            entropy_contribution=entropy_contribution if not success else 0.0
        )
        
        self.validation_events.append(event)
        
        # Update token state
        if success:
            self.update_trust(event.delta_tau)
            self.update_uncertainty(-0.01)  # Reduce uncertainty
        else:
            self.update_trust(event.delta_tau)
            self.update_entropy(event.entropy_contribution)
            self.update_uncertainty(0.01)  # Increase uncertainty
        
        # Update temporal metrics
        self.L_t = self.calculate_trust_leverage()
        self.last_updated = epoch
        
        return event
    
    def calculate_validator_reward(self, validator_id: str) -> float:
        """
        Calculate validator reward based on their contributions.
        
        R_i ∝ Δτ_i / γ_i
        
        Args:
            validator_id: Validator to calculate reward for
        
        Returns:
            Reward weight
        """
        validator_events = [e for e in self.validation_events if e.validator_id == validator_id]
        
        if not validator_events:
            return 0.0
        
        total_delta_tau = sum(e.delta_tau for e in validator_events if e.success)
        
        if self.gamma_t == 0:
            return total_delta_tau
        
        return total_delta_tau / self.gamma_t
    
    def get_compounded_understanding(self) -> float:
        """
        Calculate total compounded understanding.
        
        U_n = U_0 ∏(1 + F_i * L_i)
        """
        # Simplified: use current F_t and L_t
        # In practice, would integrate over all epochs
        return self.U_n * (1 + self.F_t * self.L_t)
    
    def is_stable(self, F_min: float = 0.03) -> bool:
        """
        Check if token meets stability conditions.
        
        α_t + β_t > γ_t and τ_t > τ_eq
        """
        # Simplified stability check
        # In practice, would use actual α_t, β_t values
        alpha_plus_beta = self.F_t * self.L_t  # Approximation
        condition1 = alpha_plus_beta > self.gamma_t
        condition2 = self.trust_tau > self.tau_eq
        condition3 = self.F_t > F_min
        
        return condition1 and condition2 and condition3
    
    def get_validation_count(self) -> int:
        """Get total number of validation events"""
        return len(self.validation_events)
    
    def get_success_rate(self) -> float:
        """Get success rate of validations"""
        if not self.validation_events:
            return 0.0
        successes = sum(1 for e in self.validation_events if e.success)
        return successes / len(self.validation_events)
    
    def get_current_value(self, k: float = 1.0) -> float:
        """
        Calculate current informational value.
        
        P_t = k * (τ_t / σ_t)
        """
        if self.uncertainty_sigma == 0:
            return float('inf')
        return k * (self.trust_tau / self.uncertainty_sigma)


def create_mathlete_token(claim_hash: str, pod_id: str, 
                         initial_trust: float = 0.5,
                         initial_uncertainty: float = 0.1,
                         initial_entropy: float = 0.05) -> MathleteToken:
    """
    Factory function to create a new Mathlete Token.
    
    Args:
        claim_hash: IPFS/Arweave hash of research artifact
        pod_id: Domain identifier
        initial_trust: Starting trust coefficient
        initial_uncertainty: Starting uncertainty
        initial_entropy: Starting entropy rate
    
    Returns:
        New MathleteToken instance
    """
    import time
    current_epoch = int(time.time())
    
    return MathleteToken(
        claim_hash=claim_hash,
        uncertainty_sigma=initial_uncertainty,
        trust_tau=initial_trust,
        gamma_t=initial_entropy,
        pod_id=pod_id,
        minted_at=current_epoch,
        last_updated=current_epoch
    )


def hash_claim(content: str) -> str:
    """
    Generate hash for a research claim.
    
    Args:
        content: Research artifact content (code, paper, etc.)
    
    Returns:
        SHA-256 hash (simplified; in practice would use IPFS)
    """
    return hashlib.sha256(content.encode()).hexdigest()


if __name__ == "__main__":
    # Example: Create MLT for Proximity Fund simulation
    print("=" * 60)
    print("MATHLETE TOKEN EXAMPLE")
    print("=" * 60)
    print()
    
    # Hash the Proximity Fund model
    import os
    model_path = os.path.join(os.path.dirname(__file__), "prove_model.py")
    with open(model_path, "r") as f:
        model_content = f.read()
    
    claim_hash = hash_claim(model_content)
    
    # Create MLT
    mlt = create_mathlete_token(
        claim_hash=claim_hash,
        pod_id="temporal_economics",
        initial_trust=0.6,
        initial_uncertainty=0.1,
        initial_entropy=0.05
    )
    
    print(f"Created MLT for Proximity Fund Model")
    print(f"Claim Hash: {claim_hash[:16]}...")
    print(f"Pod ID: {mlt.pod_id}")
    print(f"Initial Trust: {mlt.trust_tau:.2f}")
    print(f"Initial Uncertainty: {mlt.uncertainty_sigma:.2f}")
    print(f"Initial Value: {mlt.calculate_initial_value():.2f}")
    print()
    
    # Simulate validations
    print("Simulating validations...")
    for i in range(5):
        success = i < 4  # First 4 succeed, last fails
        mlt.add_validation(
            validator_id=f"validator_{i}",
            epoch=mlt.minted_at + i * 100,
            success=success,
            proof_hash=f"proof_{i}",
            delta_tau=0.1 if success else 0.0,
            entropy_contribution=0.02 if not success else 0.0
        )
        print(f"  Validation {i+1}: {'✓ Success' if success else '✗ Failure'}")
        print(f"    Trust: {mlt.trust_tau:.3f}, Uncertainty: {mlt.uncertainty_sigma:.3f}")
    
    print()
    print(f"Final State:")
    print(f"  Trust: {mlt.trust_tau:.3f}")
    print(f"  Uncertainty: {mlt.uncertainty_sigma:.3f}")
    print(f"  Entropy: {mlt.gamma_t:.3f}")
    print(f"  Trust Leverage: {mlt.calculate_trust_leverage():.3f}")
    print(f"  Validation Count: {mlt.get_validation_count()}")
    print(f"  Success Rate: {mlt.get_success_rate():.1%}")
    print(f"  Current Value: {mlt.get_current_value():.2f}")
    print(f"  Is Stable: {mlt.is_stable()}")

