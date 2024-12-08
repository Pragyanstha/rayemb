import React, { useEffect, useState } from "react";
import CircularProgress from '@mui/material/CircularProgress';

interface HeatmapProps {
    imageUrl: string;
    heatmapUrl: string;
    isLoading: boolean;
}

const Heatmap = ({ imageUrl, heatmapUrl, isLoading }: HeatmapProps) => {

    return (
        <div style={{ flexBasis: "50%", height: "100%", display: "flex", justifyContent: "center", alignItems: "center", position: 'relative' }}>
            {isLoading && (
                <div style={{
                    position: 'absolute',
                    top: 'calc(35% - 250px)',
                    left: 'calc(50% - 250px)',
                    width: '512px',
                    height: '512px',
                    backgroundColor: 'rgba(0, 0, 0, 0.5)',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    zIndex: 10
                }}>
                    <CircularProgress 
                    color="secondary"
                    size={100}
                    />
                </div>
            )}  
            {imageUrl ? (
                <React.Fragment>
                        <img src={imageUrl} alt="Base Image" style={{ display: 'block', border: '1px solid black', position: 'absolute', top: 'calc(35% - 250px)', left: 'calc(50% - 250px)' }} />
                    {heatmapUrl ? (
                        <img src={heatmapUrl} alt="Heatmap" style={{ display: 'block', position: 'absolute', top: 'calc(35% - 250px)', left: 'calc(50% - 250px)', opacity: 0.5 }} />
                    ) : (
                        <p>No heatmap</p>
                    )}
                    <div style={{ position: 'absolute', bottom: '2%', left: 'calc(50% - 250px)', color: 'white', fontFamily: 'Arial', fontSize: '16px', width: '512px', border: '1px solid white', padding: '20px', borderRadius: '10px'}}>
                        <b style={{ fontSize: '20px' }}>How to use</b> <br /> <br />
                        <ul style={{ paddingLeft: '20px' }}>
                            <li>Click inside the slice views to get its heatmap.</li>
                            <li>Scroll inside the slice view to change the slice.</li>
                            <li>Zoom in and out using the mouse wheel in the rendering view.</li>
                            <li>Blue point is the predicted projection point.</li>
                            <li>Red point is the ground truth projection point.</li>
                        </ul>
                    </div>
                </React.Fragment>
                ) : (
                <p>Loading...</p>
            )}
        </div>
    );
};

export default Heatmap;