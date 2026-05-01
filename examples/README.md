# TRACT Example Files

Sample framework documents for testing `tract prepare`.

## Usage

Prepare the CSV example:
```
tract prepare --file sample_framework.csv --framework-id example_fw --name "Example Framework"
```

Prepare the Markdown example:
```
tract prepare --file sample_framework.md --framework-id example_fw --name "Example Framework"
```

Then validate the output:
```
tract validate --file example_fw_prepared.json
```

And ingest into the crosswalk database:
```
tract ingest --file example_fw_prepared.json
```

## File Formats

- **CSV**: Standard comma-separated with `control_id`, `title`, `description` columns
- **Markdown**: One `## ID: Title` heading per control, description in the body
