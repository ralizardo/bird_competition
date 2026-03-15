"""
Query iNaturalist API for geographic priors based on occurrence data.

Pantanal bounding box:
- Latitude: -21.6 to -16.5
- Longitude: -57.6 to -55.9
"""

import csv
import json
import time
import requests
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

# Pantanal bounding box
PANTANAL_BBOX = {
    "swlat": -21.6,
    "swlng": -57.6,
    "nelat": -16.5,
    "nelng": -55.9,
}

# Expanded search radius (300km from center for broader coverage)
PANTANAL_CENTER = {"lat": -19.05, "lng": -56.75}
SEARCH_RADIUS_KM = 300

# iNaturalist API base URL
INAT_API_BASE = "https://api.inaturalist.org/v1"

# Rate limiting
REQUEST_DELAY = 1.0  # seconds between requests (be nice to the API)


@dataclass
class SpeciesGeoPrior:
    primary_label: str
    inat_taxon_id: int
    scientific_name: str
    common_name: str
    class_name: str

    # iNaturalist metrics
    inat_pantanal_obs: int = 0
    inat_brazil_obs: int = 0
    inat_global_obs: int = 0
    inat_pantanal_ratio: float = 0.0
    inat_brazil_ratio: float = 0.0

    # Query status
    query_success: bool = False
    error_message: str = ""


def load_taxonomy(taxonomy_path: str) -> list[dict]:
    """Load taxonomy.csv and filter queryable species."""
    with open(taxonomy_path, 'r') as f:
        reader = csv.DictReader(f)
        species = list(reader)

    # Filter out problematic species
    queryable = []
    skipped = []

    for s in species:
        # Skip sonotypes (unidentified insects)
        if 'son' in s['primary_label']:
            skipped.append((s['primary_label'], "sonotype"))
            continue

        # Skip domestic/cosmopolitan species
        domestic_names = [
            'Equus caballus', 'Canis familiaris', 'Bos taurus',
            'Gallus gallus', 'Passer domesticus'
        ]
        if s['scientific_name'] in domestic_names:
            skipped.append((s['primary_label'], "domestic"))
            continue

        queryable.append(s)

    print(f"Loaded {len(species)} species")
    print(f"Queryable: {len(queryable)}")
    print(f"Skipped: {len(skipped)}")

    return queryable, skipped


def query_observation_count(
    taxon_id: int,
    place_id: Optional[int] = None,
    bbox: Optional[dict] = None,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    radius: Optional[int] = None,
) -> int:
    """
    Query iNaturalist for observation count.

    Args:
        taxon_id: iNaturalist taxon ID
        place_id: iNaturalist place ID (e.g., 6878 for Brazil)
        bbox: Bounding box dict with swlat, swlng, nelat, nelng
        lat, lng, radius: Circle search parameters

    Returns:
        Total observation count
    """
    url = f"{INAT_API_BASE}/observations"

    params = {
        "taxon_id": taxon_id,
        "per_page": 0,  # We only need the count
        "verifiable": "true",  # Only research-grade or needs ID
    }

    if place_id:
        params["place_id"] = place_id

    if bbox:
        params.update(bbox)

    if lat and lng and radius:
        params["lat"] = lat
        params["lng"] = lng
        params["radius"] = radius

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("total_results", 0)
    except requests.RequestException as e:
        raise Exception(f"API request failed: {e}")


def query_species_geo_data(species: dict) -> SpeciesGeoPrior:
    """Query all geographic data for a single species."""

    result = SpeciesGeoPrior(
        primary_label=species['primary_label'],
        inat_taxon_id=int(species['inat_taxon_id']),
        scientific_name=species['scientific_name'],
        common_name=species['common_name'],
        class_name=species['class_name'],
    )

    try:
        taxon_id = result.inat_taxon_id

        # Query 1: Global observations
        result.inat_global_obs = query_observation_count(taxon_id)
        time.sleep(REQUEST_DELAY)

        # Query 2: Brazil observations (place_id=6878)
        result.inat_brazil_obs = query_observation_count(taxon_id, place_id=6878)
        time.sleep(REQUEST_DELAY)

        # Query 3: Pantanal region (bounding box)
        result.inat_pantanal_obs = query_observation_count(taxon_id, bbox=PANTANAL_BBOX)
        time.sleep(REQUEST_DELAY)

        # Calculate ratios
        if result.inat_global_obs > 0:
            result.inat_pantanal_ratio = result.inat_pantanal_obs / result.inat_global_obs
            result.inat_brazil_ratio = result.inat_brazil_obs / result.inat_global_obs

        result.query_success = True

    except Exception as e:
        result.error_message = str(e)
        result.query_success = False

    return result


def save_results(results: list[SpeciesGeoPrior], output_path: str):
    """Save results to CSV."""
    if not results:
        return

    fieldnames = list(asdict(results[0]).keys())

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"Saved {len(results)} results to {output_path}")


def save_checkpoint(results: list[SpeciesGeoPrior], checkpoint_path: str):
    """Save intermediate checkpoint as JSON."""
    with open(checkpoint_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def load_checkpoint(checkpoint_path: str) -> list[SpeciesGeoPrior]:
    """Load from checkpoint if exists."""
    path = Path(checkpoint_path)
    if not path.exists():
        return []

    with open(checkpoint_path, 'r') as f:
        data = json.load(f)

    return [SpeciesGeoPrior(**d) for d in data]


def main():
    import sys

    # Paths
    base_dir = Path(__file__).parent.parent.parent
    taxonomy_path = base_dir / "data" / "raw" / "taxonomy.csv"
    output_path = base_dir / "data" / "processed" / "inat_geo_priors.csv"
    checkpoint_path = base_dir / "data" / "processed" / "inat_geo_priors_checkpoint.json"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load taxonomy
    queryable, skipped = load_taxonomy(taxonomy_path)

    # Load checkpoint if exists
    completed_results = load_checkpoint(checkpoint_path)
    completed_labels = {r.primary_label for r in completed_results}

    print(f"\nResuming from checkpoint: {len(completed_results)} already completed", flush=True)

    # Filter out already completed
    remaining = [s for s in queryable if s['primary_label'] not in completed_labels]
    print(f"Remaining to query: {len(remaining)}", flush=True)

    if not remaining:
        print("All species already queried!")
        save_results(completed_results, output_path)
        return

    # Query each species
    results = list(completed_results)

    for i, species in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}] Querying {species['common_name']} ({species['scientific_name']})...", flush=True)

        result = query_species_geo_data(species)
        results.append(result)

        if result.query_success:
            print(f"  Global: {result.inat_global_obs:,} | Brazil: {result.inat_brazil_obs:,} | Pantanal: {result.inat_pantanal_obs:,}", flush=True)
            print(f"  Pantanal ratio: {result.inat_pantanal_ratio:.4f}", flush=True)
        else:
            print(f"  ERROR: {result.error_message}", flush=True)

        # Save checkpoint every 10 species
        if (i + 1) % 10 == 0:
            save_checkpoint(results, checkpoint_path)
            print(f"  [Checkpoint saved: {len(results)} species]", flush=True)

    # Final save
    save_checkpoint(results, checkpoint_path)
    save_results(results, output_path)

    # Summary
    successful = [r for r in results if r.query_success]
    failed = [r for r in results if not r.query_success]

    print(f"\n=== SUMMARY ===")
    print(f"Total queried: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        with_pantanal = [r for r in successful if r.inat_pantanal_obs > 0]
        print(f"Species with Pantanal observations: {len(with_pantanal)}")

        # Top species by Pantanal ratio
        print("\nTop 10 species by Pantanal concentration:")
        sorted_by_ratio = sorted(successful, key=lambda x: x.inat_pantanal_ratio, reverse=True)[:10]
        for r in sorted_by_ratio:
            print(f"  {r.common_name}: {r.inat_pantanal_ratio:.4f} ({r.inat_pantanal_obs}/{r.inat_global_obs})")


if __name__ == "__main__":
    main()
