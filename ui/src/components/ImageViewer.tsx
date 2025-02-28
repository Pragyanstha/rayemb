import React, { useEffect, useState } from "react";
import CircularProgress from '@mui/material/CircularProgress';
import useImageViewer from "../hooks/useImageViewer";

interface ImageViewProps {
    imageUrl: string;
    alt: string;
    style?: React.CSSProperties;
}

const ImageView: React.FC<ImageViewProps> = ({ imageUrl, alt, style }) => (
    <img 
        src={imageUrl} 
        alt={alt} 
        style={{ 
            display: 'block', 
            border: '1px solid black', 
            width: '90%',
            height: 'auto',
            aspectRatio: '1/1',
            ...style
        }} 
    />
);

interface HeatmapOverlayProps {
    heatmapUrl: string | null;
    isLoading: boolean;
}

const HeatmapOverlay: React.FC<HeatmapOverlayProps> = ({ heatmapUrl, isLoading }) => (
    <>
        {isLoading && (
            <div style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                width: '90%',
                aspectRatio: '1/1',
                backgroundColor: 'rgba(0, 0, 0, 0.5)',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                zIndex: 10
            }}>
                <CircularProgress color="secondary" size={100} />
            </div>
        )}
        {heatmapUrl ? (
            <img 
                src={heatmapUrl} 
                alt="Heatmap" 
                style={{ 
                    display: 'block', 
                    position: 'absolute',
                    width: '90%',
                    height: 'auto',
                    aspectRatio: '1/1',
                    opacity: 0.5 
                }} 
            />
        ) : (
            <p>No heatmap</p>
        )}
    </>
);

interface ImageContainerProps {
    children: React.ReactNode;
}

const ImageContainer: React.FC<ImageContainerProps> = ({ children }) => (
    <div style={{ 
        flex: 1, 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        position: 'relative' ,
    }}>
        {children}
    </div>
);

interface ConfigurationBoxProps {
    // Add props as needed for file upload functionality
    uploadImage: (file: File) => void;
}

const ConfigurationBox: React.FC<ConfigurationBoxProps> = ({ uploadImage }) => (
    <div style={{
        padding: '1em',
        borderRadius: '4px',
        backgroundColor: '#5a5a5a',
        color: 'white',
        fontFamily: 'monospace',
        width: '100%',
        height: '20em'
    }}>
        <h3>Configuration</h3>
        <p
            style={{
                marginTop: '1em',
                marginBottom: '0.2em'
            }}
        >Upload your X-Ray image to get started.</p>
        <input
            type="file"
            accept="image/*"
        />
        <button 
            style={{
                borderRadius: '10%',
                backgroundColor: 'grey',
                color: 'white',
                padding: '0.5em 1em',
                border: 'none',
                cursor: 'pointer',
                transition: 'background-color 0.3s'
            }}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'darkgrey'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'grey'}
            onClick={() => {
                const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
                if (fileInput && fileInput.files?.[0]) {
                    uploadImage(fileInput.files[0]);
                }
            }}
        >Upload</button>
        {/* Add more configuration options here */}
    </div>
);

const ImageViewer: React.FC<{ 
    imageUrl: string | null; 
    heatmapUrl: string | null; 
    isLoading: boolean;
    uploadImage: (file: File) => void;
}> = ({ 
    imageUrl, 
    heatmapUrl, 
    isLoading,
    uploadImage
}) => {
    return (
        <div style={{ 
            display: "flex",
            flexDirection: "column",
            width: "50%",
            gap: '1em',
        }}>
            {/* Images container */}
            <div style={{ 
                display: "flex", 
                width: "100%",
                position: 'relative',
                gap: '1em',
                flex: 1
            }}>
                {imageUrl ? (
                    <React.Fragment>
                        {/* Left side - Original Image */}
                        <ImageContainer>
                            <ImageView imageUrl={imageUrl} alt="Original Image" />
                        </ImageContainer>

                        {/* Right side - Image with Heatmap */}
                        <ImageContainer>
                            <ImageView 
                                imageUrl={imageUrl} 
                                alt="Base Image" 
                                style={{ position: 'absolute' }} 
                            />
                            <HeatmapOverlay heatmapUrl={heatmapUrl} isLoading={isLoading} />
                        </ImageContainer>
                    </React.Fragment>
                ) : (
                    <p>Loading...</p>
                )}
            </div>

            {/* Configuration box - now positioned at bottom */}
            <div style={{ 
                width: "100%", 
                padding: "0 1em",
                marginTop: 'auto'
            }}>
                <ConfigurationBox uploadImage={uploadImage} />
            </div>
        </div>
    );
};

export default ImageViewer;