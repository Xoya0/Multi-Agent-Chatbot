import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class AgentProfile:
    """Represents an AI agent's personality and configuration."""
    id: str
    name: str
    description: str
    traits: List[str]
    avatar: str
    behavior_params: Dict[str, Union[float, int]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the agent profile to a dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentProfile':
        """Create an AgentProfile instance from a dictionary.
        
        Args:
            data: Dictionary containing profile data.
            
        Returns:
            AgentProfile instance.
            
        Raises:
            ValueError: If required fields are missing or invalid.
        """
        required_fields = {'id', 'name', 'description', 'traits', 'avatar', 'behavior_params'}
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        if not isinstance(data['traits'], list):
            raise ValueError("'traits' must be a list")
        
        if not isinstance(data['behavior_params'], dict):
            raise ValueError("'behavior_params' must be a dictionary")
            
        return cls(**data)

class AgentManager:
    """Manages agent profiles and configurations."""
    
    def __init__(self, profiles_path: Optional[str] = None):
        """Initialize the agent manager.
        
        Args:
            profiles_path: Path to the JSON file containing agent profiles.
                          If None, default profiles will be used.
        """
        self.profiles: Dict[str, AgentProfile] = {}
        self.profiles_path = Path(profiles_path) if profiles_path else None
        self._load_profiles()
    
    def _load_profiles(self) -> None:
        """Load agent profiles from file or create defaults."""
        if self.profiles_path and self.profiles_path.exists():
            try:
                self._load_profiles_from_file()
            except Exception as e:
                logging.error(f"Failed to load profiles from file: {e}")
                self._create_default_profiles()
        else:
            self._create_default_profiles()
    
    def _load_profiles_from_file(self) -> None:
        """Load and validate agent profiles from a JSON file."""
        try:
            with open(self.profiles_path, 'r', encoding='utf-8') as f:
                profiles_data = json.load(f)
            
            if not isinstance(profiles_data, list):
                raise ValueError("Profile data must be a list")
            
            valid_profiles = []
            for profile_data in profiles_data:
                try:
                    profile = AgentProfile.from_dict(profile_data)
                    valid_profiles.append(profile)
                except Exception as e:
                    logging.error(f"Invalid profile data: {e}")
            
            if not valid_profiles:
                raise ValueError("No valid profiles found in file")
            
            self.profiles = {profile.id: profile for profile in valid_profiles}
            logging.info(f"Loaded {len(self.profiles)} agent profiles")
            
        except Exception as e:
            raise RuntimeError(f"Error loading profiles: {e}")
    
    def _create_default_profiles(self) -> None:
        """Create and initialize default agent profiles."""
        default_profiles = [
            AgentProfile(
                id="creative",
                name="Creative Artist",
                description="An artistic and creative personality who thinks outside the box",
                traits=[
                    "I'm passionate about art and creativity",
                    "I find joy in expressing myself through painting",
                    "I see beauty in unexpected places",
                    "I believe in the power of imagination"
                ],
                avatar="🎨",
                behavior_params={
                    "temperature": 0.9,
                    "top_p": 0.92,
                    "repetition_penalty": 1.2
                }
            ),
            AgentProfile(
                id="tech",
                name="Tech Enthusiast",
                description="A technology-focused personality who loves innovation",
                traits=[
                    "I'm fascinated by technology and innovation",
                    "I enjoy exploring new programming languages",
                    "I'm always curious about how things work",
                    "I believe technology can solve many world problems"
                ],
                avatar="💻",
                behavior_params={
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1
                }
            ),
            AgentProfile(
                id="empathetic",
                name="Empathetic Friend",
                description="A warm and understanding personality who connects with people",
                traits=[
                    "I'm a people person who loves meaningful conversations",
                    "I enjoy hearing others' stories",
                    "I'm empathetic and understanding",
                    "I believe in the power of human connection"
                ],
                avatar="🤗",
                behavior_params={
                    "temperature": 0.85,
                    "top_p": 0.92,
                    "repetition_penalty": 1.2
                }
            ),
            AgentProfile(
                id="analytical",
                name="Analytical Thinker",
                description="A logical and analytical personality who examines problems carefully",
                traits=[
                    "I approach problems with careful analysis",
                    "I enjoy intellectual discussions and debates",
                    "I value logic and reason in decision making",
                    "I'm detail-oriented and thorough"
                ],
                avatar="🧠",
                behavior_params={
                    "temperature": 0.7,
                    "top_p": 0.85,
                    "repetition_penalty": 1.1
                }
            )
        ]
        
        self.profiles = {profile.id: profile for profile in default_profiles}
        logging.info(f"Created {len(self.profiles)} default agent profiles")
        
        if self.profiles_path:
            self.save_profiles()
    
    def get_all_profiles(self) -> List[AgentProfile]:
        """Get all available agent profiles.
        
        Returns:
            List of all agent profiles.
        """
        return list(self.profiles.values())
    
    def get_profile(self, agent_id: str) -> Optional[AgentProfile]:
        """Get an agent profile by ID.
        
        Args:
            agent_id: The ID of the agent profile to retrieve.
            
        Returns:
            The agent profile if found, None otherwise.
        """
        return self.profiles.get(agent_id)
    
    def add_profile(self, profile: AgentProfile) -> None:
        """Add a new agent profile.
        
        Args:
            profile: The agent profile to add.
            
        Raises:
            ValueError: If a profile with the same ID already exists.
        """
        if profile.id in self.profiles:
            raise ValueError(f"Profile with ID '{profile.id}' already exists")
        
        self.profiles[profile.id] = profile
        if self.profiles_path:
            self.save_profiles()
    
    def update_profile(self, profile: AgentProfile) -> bool:
        """Update an existing agent profile.
        
        Args:
            profile: The updated agent profile.
            
        Returns:
            True if the profile was updated, False if it doesn't exist.
        """
        if profile.id in self.profiles:
            self.profiles[profile.id] = profile
            if self.profiles_path:
                self.save_profiles()
            return True
        return False
    
    def delete_profile(self, agent_id: str) -> bool:
        """Delete an agent profile.
        
        Args:
            agent_id: The ID of the agent profile to delete.
            
        Returns:
            True if the profile was deleted, False if it doesn't exist.
        """
        if agent_id in self.profiles:
            del self.profiles[agent_id]
            if self.profiles_path:
                self.save_profiles()
            return True
        return False
    
    def save_profiles(self) -> None:
        """Save all agent profiles to the JSON file.
        
        Raises:
            RuntimeError: If saving profiles fails.
        """
        if not self.profiles_path:
            logging.error("No profiles path specified for saving")
            return

        backup_path = self.profiles_path.with_suffix('.json.bak')
        temp_path = self.profiles_path.with_suffix('.json.tmp')
        
        try:
            # Ensure directory exists
            self.profiles_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first
            profiles_data = [profile.to_dict() for profile in self.profiles.values()]
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, indent=2, ensure_ascii=False)
            
            # Create backup of existing file if it exists
            if self.profiles_path.exists():
                self.profiles_path.replace(backup_path)
            
            # Rename temporary file to actual file
            temp_path.replace(self.profiles_path)
            
            # Remove backup file on successful save
            if backup_path.exists():
                backup_path.unlink()
                
            logging.info(f"Saved {len(self.profiles)} agent profiles")
            
        except Exception as e:
            # Restore from backup if available
            if backup_path.exists() and not self.profiles_path.exists():
                backup_path.replace(self.profiles_path)
                logging.info("Restored profile file from backup")
            
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
                
            raise RuntimeError(f"Failed to save profiles: {e}")

# Create a singleton instance of the agent manager
agent_manager = AgentManager(profiles_path=str(Path(__file__).parent / "agent_profiles.json"))