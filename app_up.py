from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
from pathlib import Path
import concurrent.futures
from PIL import Image, ExifTags
import imagehash
import json
import csv
from contextlib import asynccontextmanager
import asyncio
import uuid
import logging
from logging.handlers import RotatingFileHandler
import io
import shutil
import psutil
from collections import defaultdict
import glob
from datetime import timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global variables
image_hashes = []
processing_status = {}
processing_results = {}
duplicate_results = {}


# Custom Exceptions
class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass


class DirectoryNotFoundError(ImageProcessingError):
    pass


class PermissionError(ImageProcessingError):
    pass


# Pydantic models
class ProcessingRequest(BaseModel):
    base_directory: str
    dates: List[str]
    school_codes: List[str]
    classes: List[str] = Field(default=["ECE", "Nursery", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5"])
    max_workers: int = Field(default=4, ge=1, le=16)


class ProcessingStatus(BaseModel):
    job_id: str
    status: str
    progress: Dict[str, Any]
    start_time: datetime
    current_task: Optional[str] = None
    estimated_completion: Optional[datetime] = None


class SchoolClassRequest(BaseModel):
    school_code: str
    class_name: str
    base_directory: str
    dates: List[str]
    max_workers: int = 4


class HashStats(BaseModel):
    total_images: int
    unique_hashes: int
    total_size_mb: float
    total_size_gb: float
    schools_processed: int
    classes_processed: int


class DuplicateGroup(BaseModel):
    hash_value: str
    count: int
    items: List[Dict[str, Any]]
    wasted_space_mb: float
    wasted_space_gb: float
    dates: List[str]
    is_cross_date: bool


class SchoolClassDuplicates(BaseModel):
    school_code: str
    class_name: str
    duplicate_groups: List[DuplicateGroup]
    total_duplicates: int
    wasted_space_mb: float
    wasted_space_gb: float


class BulkProcessingRequest(BaseModel):
    requests: List[ProcessingRequest]


# Helper functions
def hash_file(file):
    try:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")

        if not os.access(file, os.R_OK):
            raise PermissionError(f"Permission denied: {file}")

        hashes = []
        img = Image.open(file)

        file_size = get_file_size(file)
        image_size = get_image_size(img)
        capture_time = get_capture_time(img)

        for angle in [0, 90, 180, 270]:
            if angle > 0:
                turned_img = img.rotate(angle, expand=True)
            else:
                turned_img = img
            hashes.append(str(imagehash.phash(turned_img)))

        hashes = ''.join(sorted(hashes))

        return file, hashes, file_size, image_size, capture_time
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Access error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing {file}: {e}")
        raise ImageProcessingError(f"Failed to process {file}: {str(e)}")


def get_file_size(file_name):
    try:
        return os.path.getsize(file_name)
    except FileNotFoundError:
        return 0


def get_image_size(img):
    return "{} x {}".format(*img.size)


def get_capture_time(img):
    try:
        exif = {
            ExifTags.TAGS[k]: v
            for k, v in img._getexif().items()
            if k in ExifTags.TAGS
        }
        return exif.get("DateTimeOriginal", "Time unknown")
    except:
        return "Time unknown"


def get_image_files(directory_path):
    """Get all image files from the specified directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_files = []

    directory = Path(directory_path)
    if not directory.exists():
        print(f"Directory {directory_path} does not exist!")
        return []

    try:
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in image_extensions and file_path.is_file():
                image_files.append(str(file_path))

    except PermissionError as e:
        print(f"Permission denied accessing {directory_path}: {e}")
    except Exception as e:
        print(f"Error scanning directory {directory_path}: {e}")

    return image_files


async def process_school_class_background(job_id: str, school_code: str, class_name: str,
                                          base_directory: str, dates: List[str], max_workers: int):
    """Background task to process a specific school and class"""
    try:
        processing_status[job_id]["progress"]["current_school"] = school_code
        processing_status[job_id]["progress"]["current_class"] = class_name

        # Construct the full directory path
        directory_path = os.path.join(base_directory, school_code, "class_images", class_name)

        all_results = []

        for date in dates:
            date_directory = os.path.join(directory_path, date)

            # Update status
            processing_status[job_id]["progress"]["current_date"] = date
            processing_status[job_id]["current_task"] = f"Processing {school_code}/{class_name}/{date}"

            # Get all image files for this date
            image_files = get_image_files(date_directory)
            if not image_files:
                continue

            # Process images in parallel
            date_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(hash_file, file): file for file in image_files}

                for future in concurrent.futures.as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        result = future.result()
                        if result is not None:
                            file_path, hash_value, file_size, image_size, capture_time = result
                            result_dict = {
                                'file_path': file_path,
                                'hash_value': hash_value,
                                'file_size': file_size,
                                'image_size': image_size,
                                'capture_time': capture_time,
                                'date': date,
                                'directory': date_directory,
                                'class_name': class_name,
                                'school_code': school_code
                            }
                            date_results.append(result_dict)

                            # Update progress
                            processing_status[job_id]["progress"]["images_processed"] += 1
                    except Exception as exc:
                        print(f"File {file} generated an exception: {exc}")

            all_results.extend(date_results)

        # Store results
        processing_results[job_id].extend(all_results)

        # Update global hashes
        global image_hashes
        for result in all_results:
            image_hashes.append((
                result['file_path'], result['hash_value'], result['file_size'],
                result['image_size'], result['capture_time'], result['date'],
                result['class_name'], result['school_code']
            ))

        return all_results

    except Exception as e:
        print(f"Error processing {school_code}/{class_name}: {e}")
        raise


async def find_duplicates_for_job(job_id: str):
    """Find duplicates for a specific job"""
    try:
        results = processing_results.get(job_id, [])
        if not results:
            return {}

        # Group by school, class, and hash
        school_groups = {}
        for result in results:
            school_code = result['school_code']
            class_name = result['class_name']
            hash_value = result['hash_value']

            if school_code not in school_groups:
                school_groups[school_code] = {}

            if class_name not in school_groups[school_code]:
                school_groups[school_code][class_name] = {}

            if hash_value not in school_groups[school_code][class_name]:
                school_groups[school_code][class_name][hash_value] = []

            school_groups[school_code][class_name][hash_value].append(result)

        # Find duplicates
        school_class_duplicates = {}
        for school_code, class_groups in school_groups.items():
            school_class_duplicates[school_code] = {}
            for class_name, hash_groups in class_groups.items():
                duplicates = []
                for hash_value, items in hash_groups.items():
                    if len(items) > 1:
                        # Sort by file size (largest first)
                        sorted_items = sorted(items, key=lambda x: x['file_size'], reverse=True)

                        # Calculate wasted space
                        wasted_space = sum(item['file_size'] for item in sorted_items[1:])

                        # Get unique dates
                        dates = sorted(set(item['date'] for item in items))

                        duplicates.append({
                            'hash_value': hash_value,
                            'count': len(items),
                            'items': items,
                            'wasted_space_mb': wasted_space / (1024 * 1024),
                            'wasted_space_gb': wasted_space / (1024 * 1024 * 1024),
                            'dates': dates,
                            'is_cross_date': len(dates) > 1
                        })

                if duplicates:
                    total_wasted_space = sum(d['wasted_space_mb'] for d in duplicates)
                    school_class_duplicates[school_code][class_name] = {
                        'duplicate_groups': duplicates,
                        'total_duplicates': sum(d['count'] - 1 for d in duplicates),
                        'wasted_space_mb': total_wasted_space,
                        'wasted_space_gb': total_wasted_space / 1024
                    }

        duplicate_results[job_id] = school_class_duplicates
        return school_class_duplicates

    except Exception as e:
        print(f"Error finding duplicates: {e}")
        raise


# CSV Export Helper Functions
async def create_school_summary_csv(job_id: str, filename: str):
    """Create CSV showing schools with duplicate images and their dates"""

    # Get duplicates for this job
    duplicates = duplicate_results.get(job_id, {})
    results = processing_results.get(job_id, [])

    if not duplicates and not results:
        raise HTTPException(status_code=404, detail="No data available for export")

    # Prepare data for CSV - using a set to avoid duplicate rows for the same group
    csv_rows = set()

    if duplicates:
        # Create summary from duplicate analysis
        for school_code, class_groups in duplicates.items():
            for class_name, class_data in class_groups.items():
                for group in class_data.get('duplicate_groups', []):
                    # Get all unique dates in this duplicate group
                    dates_in_group = sorted(set(group['dates']))

                    # Create a row for this duplicate group
                    row_key = (school_code, class_name, group['count'], tuple(dates_in_group))
                    csv_rows.add(row_key)
    else:
        # Create summary from raw results if no duplicates found
        # Group by school and hash to find duplicates
        school_hashes = defaultdict(lambda: defaultdict(list))

        for result in results:
            school_code = result['school_code']
            class_name = result['class_name']
            hash_value = result['hash_value']

            school_hashes[school_code][hash_value].append(result)

        # Process duplicate groups
        for school_code, hash_groups in school_hashes.items():
            for hash_value, items in hash_groups.items():
                if len(items) > 1:
                    # Get all unique dates in this duplicate group
                    dates_in_group = sorted(set(item['date'] for item in items))

                    # Create a row for this duplicate group
                    row_key = (school_code, items[0]['class_name'], len(items), tuple(dates_in_group))
                    csv_rows.add(row_key)

    # Convert set to list of dictionaries for CSV
    csv_data = []
    for school_code, class_name, duplicate_count, dates_tuple in csv_rows:
        # Join dates into a string
        duplicate_dates = ', '.join(dates_tuple)

        csv_data.append({
            'school_code': school_code,
            'class_name': class_name,
            'duplicate_count': duplicate_count,
            'duplicate_date': duplicate_dates
        })

    # Sort by school_code, class_name, then duplicate_count (descending)
    csv_data.sort(key=lambda x: (x['school_code'], x['class_name'], -x['duplicate_count']))

    # Create CSV file
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        if csv_data:
            # Define only the requested fields
            fieldnames = ['school_code', 'class_name', 'duplicate_count', 'duplicate_date']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        else:
            # Write header even if no duplicates found
            fieldnames = ['school_code', 'class_name', 'duplicate_count', 'duplicate_date']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                'school_code': 'No duplicates found',
                'class_name': '',
                'duplicate_count': '',
                'duplicate_date': ''
            })

    return FileResponse(
        filename,
        media_type='text/csv',
        filename=filename
    )


async def create_detailed_duplicate_csv(job_id: str, filename: str):
    """Create a more detailed CSV with school-wise duplicate analysis"""

    duplicates = duplicate_results.get(job_id, {})

    if not duplicates:
        # If no duplicates found, return empty CSV with headers
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['status', 'message'])
            writer.writerow(['No duplicates found', f'Job ID: {job_id}'])

        return FileResponse(
            filename,
            media_type='text/csv',
            filename=filename
        )

    # Prepare data for CSV
    csv_data = []

    # Summary section
    total_schools = len(duplicates)
    total_classes = 0
    total_duplicate_groups = 0
    total_wasted_space_mb = 0

    for school_code, class_groups in duplicates.items():
        school_total_duplicates = 0
        school_wasted_space_mb = 0

        for class_name, class_data in class_groups.items():
            total_classes += 1
            total_duplicate_groups += len(class_data.get('duplicate_groups', []))
            school_total_duplicates += class_data.get('total_duplicates', 0)
            school_wasted_space_mb += class_data.get('wasted_space_mb', 0)

            # Add class-level summary
            csv_data.append({
                'section': 'CLASS_SUMMARY',
                'school_code': school_code,
                'class_name': class_name,
                'duplicate_groups': len(class_data.get('duplicate_groups', [])),
                'total_duplicate_images': class_data.get('total_duplicates', 0),
                'wasted_space_mb': f"{class_data.get('wasted_space_mb', 0):.2f}",
                'wasted_space_gb': f"{class_data.get('wasted_space_gb', 0):.4f}",
                'recommendation': f"Review {len(class_data.get('duplicate_groups', []))} duplicate groups"
            })

            # Add details for each duplicate group
            for group in class_data.get('duplicate_groups', []):
                for item in group['items']:
                    csv_data.append({
                        'section': 'DUPLICATE_DETAILS',
                        'school_code': school_code,
                        'class_name': class_name,
                        'group_id': group['hash_value'][:8],
                        'file_path': item['file_path'],
                        'date': item['date'],
                        'file_size_mb': f"{item['file_size'] / (1024 * 1024):.2f}",
                        'image_size': item['image_size'],
                        'duplicate_count': group['count'],
                        'dates_in_group': ', '.join(sorted(group['dates'])),
                        'cross_date': 'Yes' if group['is_cross_date'] else 'No',
                        'wasted_in_group_mb': f"{group['wasted_space_mb']:.2f}",
                        'action_recommended': 'Keep first, delete others' if group['items'].index(
                            item) == 0 else 'Delete duplicate'
                    })

        total_wasted_space_mb += school_wasted_space_mb

        # Add school-level summary
        csv_data.append({
            'section': 'SCHOOL_SUMMARY',
            'school_code': school_code,
            'classes_with_duplicates': len(class_groups),
            'total_duplicate_images': school_total_duplicates,
            'total_wasted_space_mb': f"{school_wasted_space_mb:.2f}",
            'total_wasted_space_gb': f"{school_wasted_space_mb / 1024:.4f}",
            'assessment': 'Needs cleanup' if school_wasted_space_mb > 100 else 'Minimal duplicates'
        })

    # Create CSV file with sections
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'section', 'school_code', 'class_name', 'group_id', 'file_path', 'date',
            'file_size_mb', 'image_size', 'duplicate_count', 'dates_in_group',
            'cross_date', 'wasted_in_group_mb', 'action_recommended',
            'duplicate_groups', 'total_duplicate_images', 'wasted_space_mb',
            'wasted_space_gb', 'recommendation', 'classes_with_duplicates',
            'total_wasted_space_mb', 'total_wasted_space_gb', 'assessment'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write overall summary first
        writer.writerow({
            'section': 'OVERALL_SUMMARY',
            'school_code': f'Job ID: {job_id}',
            'total_schools': total_schools,
            'total_classes': total_classes,
            'total_duplicate_groups': total_duplicate_groups,
            'total_wasted_space_mb': f"{total_wasted_space_mb:.2f}",
            'total_wasted_space_gb': f"{total_wasted_space_mb / 1024:.4f}"
        })

        writer.writerow({})  # Empty row for separation

        # Write all data rows
        writer.writerows(csv_data)

    return FileResponse(
        filename,
        media_type='text/csv',
        filename=filename
    )


# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting FastAPI server...")
    yield
    # Shutdown
    logger.info("Shutting down FastAPI server...")


# Create FastAPI app
app = FastAPI(
    title="Image Duplicate Detection API",
    description="API for detecting duplicate images across schools and classes",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Background task functions
async def process_images_background(job_id: str, base_directory: str, dates: List[str],
                                    school_codes: List[str], classes: List[str], max_workers: int):
    """Background task to process all images"""
    try:
        processing_status[job_id]["status"] = "processing"
        processing_status[job_id]["current_task"] = "Starting image processing..."

        total_schools = len(school_codes)
        total_classes = len(classes)

        for school_index, school_code in enumerate(school_codes):
            processing_status[job_id]["progress"]["current_school"] = school_code
            processing_status[job_id]["progress"]["schools_processed"] = school_index

            for class_index, class_name in enumerate(classes):
                processing_status[job_id]["progress"]["current_class"] = class_name
                processing_status[job_id]["progress"]["classes_processed"] = (
                        school_index * total_classes + class_index
                )

                processing_status[job_id]["current_task"] = (
                    f"Processing {school_code} - {class_name}"
                )

                # Process this school and class
                await process_school_class_background(
                    job_id, school_code, class_name,
                    base_directory, dates, max_workers
                )

        # Update status
        processing_status[job_id]["status"] = "finding_duplicates"
        processing_status[job_id]["current_task"] = "Finding duplicates..."

        # Find duplicates
        await find_duplicates_for_job(job_id)

        # Finalize
        processing_status[job_id]["status"] = "completed"
        processing_status[job_id]["current_task"] = "Processing complete"
        processing_status[job_id]["progress"]["schools_processed"] = total_schools
        processing_status[job_id]["progress"]["classes_processed"] = total_schools * total_classes

    except Exception as e:
        processing_status[job_id]["status"] = "failed"
        processing_status[job_id]["current_task"] = f"Error: {str(e)}"
        logger.error(f"Error in background processing job {job_id}: {e}")
        raise


async def process_school_class_job(job_id: str, request: SchoolClassRequest):
    """Background task for single school-class processing"""
    try:
        # Process the school and class
        results = await process_school_class_background(
            job_id,
            request.school_code,
            request.class_name,
            request.base_directory,
            request.dates,
            request.max_workers
        )

        # Find duplicates
        await find_duplicates_for_job(job_id)

        # Update status
        processing_status[job_id]["status"] = "completed"
        processing_status[job_id]["progress"]["schools_processed"] = 1
        processing_status[job_id]["progress"]["classes_processed"] = 1
        processing_status[job_id]["current_task"] = "Processing complete"

    except Exception as e:
        processing_status[job_id]["status"] = "failed"
        processing_status[job_id]["current_task"] = f"Error: {str(e)}"
        raise


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Return API documentation page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Duplicate Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { display: inline-block; padding: 5px 10px; border-radius: 3px; color: white; }
            .get { background: #4CAF50; }
            .post { background: #2196F3; }
            .put { background: #FF9800; }
            .delete { background: #F44336; }
            .param { background: #9C27B0; }
        </style>
    </head>
    <body>
        <h1>Image Duplicate Detection API</h1>
        <p>API for detecting duplicate images across schools and classes</p>

        <h2>Endpoints:</h2>

        <div class="endpoint">
            <span class="method get">GET</span> <strong>/status</strong> - Get API status
        </div>

        <div class="endpoint">
            <span class="method post">POST</span> <strong>/process</strong> - Process images for duplicate detection
        </div>

        <div class="endpoint">
            <span class="method get">GET</span> <strong>/process/status/{job_id}</strong> - Get processing status
        </div>

        <div class="endpoint">
            <span class="method get">GET</span> <strong>/results/{job_id}</strong> - Get processing results
        </div>

        <div class="endpoint">
            <span class="method get">GET</span> <strong>/duplicates/{job_id}</strong> - Get duplicate analysis
        </div>

        <div class="endpoint">
            <span class="method get">GET</span> <strong>/stats</strong> - Get global statistics
        </div>

        <h2>Export Options:</h2>

        <div class="endpoint">
            <span class="method get">GET</span> <strong>/export/{job_id}?format=csv</strong><br>
            <span class="param">Parameters:</span> format=csv (default) or json<br>
            Returns: Basic CSV with duplicate information
        </div>

        <div class="endpoint">
            <span class="method get">GET</span> <strong>/export/{job_id}?format=csv&detailed=true</strong><br>
            Returns: Detailed duplicate analysis with school and date information
        </div>

        <div class="endpoint">
            <span class="method get">GET</span> <strong>/export/school-dates/{job_id}</strong><br>
            Returns: School-date matrix showing duplicate distribution across dates
        </div>

        <div class="endpoint">
            <span class="method get">GET</span> <strong>/health</strong> - Health check endpoint
        </div>

        <div class="endpoint">
            <span class="method get">GET</span> <strong>/cleanup/temp-files?hours_old=24</strong> - Clean up temporary CSV files
        </div>

        <div class="endpoint">
            <span class="method post">POST</span> <strong>/process/school-class</strong> - Process single school and class
        </div>

        <div class="endpoint">
            <span class="method post">POST</span> <strong>/process/bulk</strong> - Process multiple requests in bulk
        </div>

        <div class="endpoint">
            <span class="method get">GET</span> <strong>/clear/{job_id}</strong> - Clear specific job data
        </div>

        <div class="endpoint">
            <span class="method get">GET</span> <strong>/clear/all</strong> - Clear all job data
        </div>

        <div class="endpoint">
            <span class="method get">GET</span> <strong>/preview/{job_id}/{image_id}?size=thumbnail</strong> - Preview an image
        </div>

        <div class="endpoint">
            <span class="method websocket">WS</span> <strong>/ws/progress/{job_id}</strong> - WebSocket for real-time progress updates
        </div>

        <h2>Interactive API Docs:</h2>
        <ul>
            <li><a href="/docs">Swagger UI Documentation</a></li>
            <li><a href="/redoc">ReDoc Documentation</a></li>
        </ul>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/status")
async def get_status():
    """Get API status"""
    return {
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len(processing_status),
        "total_images_processed": len(image_hashes)
    }


@app.post("/process", response_model=ProcessingStatus)
async def start_processing(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """Start image processing for duplicate detection"""
    job_id = str(uuid.uuid4())

    # Initialize job status
    processing_status[job_id] = {
        "job_id": job_id,
        "status": "initializing",
        "progress": {
            "total_schools": len(request.school_codes),
            "total_classes": len(request.classes),
            "schools_processed": 0,
            "classes_processed": 0,
            "images_processed": 0,
            "current_school": None,
            "current_class": None,
            "current_date": None
        },
        "start_time": datetime.now(),
        "current_task": "Starting processing..."
    }

    # Initialize results storage
    processing_results[job_id] = []

    # Add background task
    background_tasks.add_task(
        process_images_background,
        job_id,
        request.base_directory,
        request.dates,
        request.school_codes,
        request.classes,
        request.max_workers
    )

    return processing_status[job_id]


@app.get("/process/status/{job_id}", response_model=ProcessingStatus)
async def get_processing_status(job_id: str):
    """Get processing status for a specific job"""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")

    return processing_status[job_id]


@app.get("/results/{job_id}")
async def get_processing_results(job_id: str):
    """Get processing results for a specific job"""
    if job_id not in processing_results:
        raise HTTPException(status_code=404, detail="Job not found")

    results = processing_results[job_id]
    return {
        "job_id": job_id,
        "total_results": len(results),
        "results": results[:100],  # Return first 100 results
        "has_more": len(results) > 100
    }


@app.get("/duplicates/{job_id}")
async def get_duplicates(job_id: str, school_code: Optional[str] = None,
                         class_name: Optional[str] = None):
    """Get duplicate analysis for a specific job"""
    if job_id not in duplicate_results:
        raise HTTPException(status_code=404, detail="Duplicates not found for this job")

    duplicates = duplicate_results[job_id]

    # Filter by school and class if specified
    if school_code:
        if school_code not in duplicates:
            raise HTTPException(status_code=404, detail="School not found")
        duplicates = {school_code: duplicates[school_code]}

        if class_name:
            if class_name not in duplicates[school_code]:
                raise HTTPException(status_code=404, detail="Class not found")
            duplicates[school_code] = {class_name: duplicates[school_code][class_name]}

    # Calculate totals
    total_schools = len(duplicates)
    total_classes = sum(len(classes) for classes in duplicates.values())
    total_duplicate_groups = sum(
        len(data['duplicate_groups'])
        for school in duplicates.values()
        for data in school.values()
    )
    total_wasted_space_mb = sum(
        data['wasted_space_mb']
        for school in duplicates.values()
        for data in school.values()
    )

    return {
        "job_id": job_id,
        "summary": {
            "total_schools_with_duplicates": total_schools,
            "total_classes_with_duplicates": total_classes,
            "total_duplicate_groups": total_duplicate_groups,
            "total_wasted_space_mb": total_wasted_space_mb,
            "total_wasted_space_gb": total_wasted_space_mb / 1024
        },
        "duplicates": duplicates
    }


@app.get("/stats")
async def get_statistics():
    """Get global statistics"""
    global image_hashes

    if not image_hashes:
        return {
            "message": "No images processed yet",
            "total_images": 0,
            "unique_hashes": 0,
            "total_size_mb": 0,
            "total_size_gb": 0
        }

    total_size = sum(h[2] for h in image_hashes)
    unique_hashes = len(set(h[1] for h in image_hashes))

    # Get school and class stats
    school_stats = {}
    for file_path, hash_value, file_size, image_size, capture_time, date, class_name, school_code in image_hashes:
        if school_code not in school_stats:
            school_stats[school_code] = {}

        if class_name not in school_stats[school_code]:
            school_stats[school_code][class_name] = {'count': 0, 'size': 0}

        school_stats[school_code][class_name]['count'] += 1
        school_stats[school_code][class_name]['size'] += file_size

    return {
        "total_images": len(image_hashes),
        "unique_hashes": unique_hashes,
        "total_size_mb": total_size / (1024 * 1024),
        "total_size_gb": total_size / (1024 * 1024 * 1024),
        "school_stats": school_stats,
        "active_jobs": len(processing_status),
        "completed_jobs": len(processing_results)
    }


@app.get("/export/{job_id}")
async def export_results(job_id: str, format: str = "csv", detailed: bool = False):
    """Export results in specified format"""
    if job_id not in processing_results:
        raise HTTPException(status_code=404, detail="Job not found")

    if format.lower() == "csv":
        if detailed:
            # Create detailed duplicate analysis CSV
            filename = f"duplicate_analysis_{job_id}.csv"
            return await create_detailed_duplicate_csv(job_id, filename)
        else:
            # Create summary CSV with school-wise duplicate dates
            filename = f"school_duplicates_summary_{job_id}.csv"
            return await create_school_summary_csv(job_id, filename)

    elif format.lower() == "json":
        results = processing_results[job_id]
        return JSONResponse(
            content={
                "job_id": job_id,
                "total_results": len(results),
                "results": results
            }
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use 'csv' or 'json'.")


@app.get("/export/school-dates/{job_id}")
async def export_school_date_matrix(job_id: str):
    """Export a matrix showing schools with duplicate dates across classes"""

    if job_id not in duplicate_results:
        raise HTTPException(status_code=404, detail="No duplicates found for this job")

    duplicates = duplicate_results[job_id]

    # Create a matrix: School -> Date -> Count of duplicates
    school_date_matrix = {}

    for school_code, class_groups in duplicates.items():
        if school_code not in school_date_matrix:
            school_date_matrix[school_code] = {}

        for class_name, class_data in class_groups.items():
            for group in class_data.get('duplicate_groups', []):
                for date in group['dates']:
                    if date not in school_date_matrix[school_code]:
                        school_date_matrix[school_code][date] = 0
                    school_date_matrix[school_code][date] += group['count']

    # Create CSV
    filename = f"school_date_matrix_{job_id}.csv"

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        # Get all unique dates
        all_dates = sorted(set(
            date
            for school_data in school_date_matrix.values()
            for date in school_data.keys()
        ))

        fieldnames = ['school_code'] + all_dates + ['total_duplicates', 'total_dates']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for school_code, date_counts in school_date_matrix.items():
            row = {'school_code': school_code}
            total_duplicates = 0

            for date in all_dates:
                count = date_counts.get(date, 0)
                row[date] = count
                total_duplicates += count

            row['total_duplicates'] = total_duplicates
            row['total_dates'] = len([d for d in all_dates if date_counts.get(d, 0) > 0])

            writer.writerow(row)

    return FileResponse(
        filename,
        media_type='text/csv',
        filename=filename
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        disk_usage = shutil.disk_usage("/")
        process = psutil.Process()

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "disk_space": {
                "total_gb": disk_usage.total / (1024 ** 3),
                "used_gb": disk_usage.used / (1024 ** 3),
                "free_gb": disk_usage.free / (1024 ** 3),
                "free_percent": (disk_usage.free / disk_usage.total) * 100
            },
            "memory": {
                "rss_mb": process.memory_info().rss / (1024 ** 2),
                "vms_mb": process.memory_info().vms / (1024 ** 2)
            },
            "active_jobs": len(processing_status),
            "total_images_processed": len(image_hashes)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/cleanup/temp-files")
async def cleanup_temp_files(hours_old: int = 24):
    """Clean up temporary CSV files older than specified hours"""

    temp_files = glob.glob("*.csv")
    deleted_files = []

    cutoff_time = datetime.now() - timedelta(hours=hours_old)

    for file_path in temp_files:
        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))

        if file_mtime < cutoff_time:
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")

    return {
        "deleted_files": deleted_files,
        "total_deleted": len(deleted_files),
        "cutoff_time": cutoff_time.isoformat()
    }


@app.post("/process/school-class")
async def process_single_school_class(request: SchoolClassRequest, background_tasks: BackgroundTasks):
    """Process a single school and class"""
    job_id = str(uuid.uuid4())

    processing_status[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "progress": {
            "total_schools": 1,
            "total_classes": 1,
            "schools_processed": 0,
            "classes_processed": 0,
            "images_processed": 0,
            "current_school": request.school_code,
            "current_class": request.class_name,
            "current_date": None
        },
        "start_time": datetime.now(),
        "current_task": f"Processing {request.school_code}/{request.class_name}"
    }

    processing_results[job_id] = []

    background_tasks.add_task(
        process_school_class_job,
        job_id,
        request
    )

    return processing_status[job_id]


@app.post("/process/bulk")
async def bulk_processing(request: BulkProcessingRequest, background_tasks: BackgroundTasks):
    """Process multiple requests in bulk"""
    job_ids = []
    for req in request.requests:
        job_id = str(uuid.uuid4())

        # Initialize job status
        processing_status[job_id] = {
            "job_id": job_id,
            "status": "initializing",
            "progress": {
                "total_schools": len(req.school_codes),
                "total_classes": len(req.classes),
                "schools_processed": 0,
                "classes_processed": 0,
                "images_processed": 0,
                "current_school": None,
                "current_class": None,
                "current_date": None
            },
            "start_time": datetime.now(),
            "current_task": "Starting processing..."
        }

        # Initialize results storage
        processing_results[job_id] = []

        # Add background task
        background_tasks.add_task(
            process_images_background,
            job_id,
            req.base_directory,
            req.dates,
            req.school_codes,
            req.classes,
            req.max_workers
        )

        job_ids.append(job_id)

    return {
        "message": f"Started {len(job_ids)} bulk processing jobs",
        "job_ids": job_ids,
        "total_requests": len(job_ids)
    }


@app.get("/preview/{job_id}/{image_hash}")
async def preview_image(job_id: str, image_hash: str, size: str = "thumbnail"):
    """Get a preview of an image from a specific job"""
    if job_id not in processing_results:
        raise HTTPException(status_code=404, detail="Job not found")

    # Find image by hash in results
    results = processing_results.get(job_id, [])
    image_data = next((r for r in results if r.get('hash_value') == image_hash), None)

    if not image_data:
        raise HTTPException(status_code=404, detail="Image not found")

    file_path = image_data['file_path']

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image file not found")

    try:
        # Create thumbnail
        with Image.open(file_path) as img:
            if size == "thumbnail":
                img.thumbnail((200, 200))
            elif size == "medium":
                img.thumbnail((800, 800))

            # Convert to bytes
            img_byte_arr = io.BytesIO()
            img_format = img.format if img.format else 'JPEG'
            img.save(img_byte_arr, format=img_format)
            img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type=f"image/{img_format.lower()}")
    except Exception as e:
        logger.error(f"Error generating preview: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating preview: {str(e)}")


@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket for real-time progress updates"""
    await websocket.accept()
    try:
        while True:
            if job_id in processing_status:
                status = processing_status[job_id]
                await websocket.send_json(status)

            # Check if processing is complete
            if job_id in processing_status and processing_status[job_id]["status"] in ["completed", "failed"]:
                await websocket.send_json({
                    "status": "complete",
                    "job_status": processing_status[job_id]["status"]
                })
                break

            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        try:
            await websocket.close()
        except:
            pass


@app.get("/clear/{job_id}")
async def clear_job_data(job_id: str):
    """Clear data for a specific job"""
    if job_id in processing_status:
        del processing_status[job_id]
    if job_id in processing_results:
        del processing_results[job_id]
    if job_id in duplicate_results:
        del duplicate_results[job_id]

    return {"message": f"Job {job_id} data cleared"}


@app.get("/clear/all")
async def clear_all_data():
    """Clear all job data"""
    global image_hashes
    image_hashes.clear()
    processing_status.clear()
    processing_results.clear()
    duplicate_results.clear()

    return {"message": "All data cleared"}


@app.get("/example/request")
async def get_example_request():
    """Get example request body for processing"""
    return {
        "base_directory": "/path/to/parent/directory",
        "dates": ["2025-10-13", "2025-10-14", "2025-10-15"],
        "school_codes": ["31110073", "31110084"],
        "classes": ["ECE", "Class 1"],
        "max_workers": 4
    }


# Error handlers
@app.exception_handler(ImageProcessingError)
async def image_processing_exception_handler(request: Request, exc: ImageProcessingError):
    return JSONResponse(
        status_code=500,
        content={"message": f"Image processing error: {str(exc)}"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": f"Internal server error: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)