import React, { useEffect, useState } from "react";
import { AppBar, Toolbar, Typography, IconButton } from "@mui/material";
import GitHubIcon from '@mui/icons-material/GitHub';
import AlternateEmailIcon from '@mui/icons-material/AlternateEmail';
import Viewer from "./components/Viewer";
import Heatmap from "./components/Heatmap";
import { getImage, getHeatmap } from "./driver";
import logo from "/icon.png";

const App = () => {

  const niftiUrl = "/test.nii.gz";
  const onNodeClick = (node: number[]) => {
    setIsLoading(true);
    getHeatmap(node).then(setHeatmapUrl).then(() => setIsLoading(false));
  };
  const [imageUrl, setImageUrl] = useState<string>("");
  const [heatmapUrl, setHeatmapUrl] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  useEffect(() => {
    setIsLoading(true);
    getImage().then(setImageUrl).then(() => setIsLoading(false));
  }, []);

  return (
    <div style={{ height: "100vh", display: "flex", gap:"10px", flexDirection: "column" }}>
      <AppBar position="static" sx={{ width: "100vw", height: "60px" }}>
      <Toolbar sx={{ backgroundColor: "primary.main", justifyContent: "space-between" }}>
          <img src={logo} alt="logo" style={{ width: "25px", height: "25px" }} />
          <Typography variant="h6" style={{ marginRight: "auto", marginLeft: "10px" }}>
            <b style={{ fontSize: "25px", color: "#F8EDED", fontFamily: "Roboto" }}>RayEmb</b> <span style={{ fontSize: "14px", color: "white", fontFamily: "monospace", marginLeft: "5px" }}>Interactive Viewer</span>
          </Typography>
          <div>
            <IconButton component="a" href="https://github.com/Pragyanstha/rayemb" target="_blank" rel="noopener noreferrer">
              <GitHubIcon />
            </IconButton>
            <IconButton component="a" href="https://pragyanstha.github.io" target="_blank" rel="noopener noreferrer">
              <AlternateEmailIcon />
            </IconButton>
          </div>
        </Toolbar>
      </AppBar>
      <div style={{ flex: 1.0, maxHeight: "calc(100vh - 120px)", paddingLeft: "10px", width: "100%", display: "flex", justifyContent: "space-between", flexDirection: "row"}} >
          <Viewer niftiUrl={niftiUrl} onNodeClick={onNodeClick} />
          <Heatmap imageUrl={imageUrl} heatmapUrl={heatmapUrl} isLoading={isLoading} />
      </div>
      <AppBar position="static" color="primary" sx={{ marginBottom: "0", width: "100vw", height: "40px"}}>
      <Typography variant="h6" sx={{ height: "100%", backgroundColor: "primary.main", textAlign: "right", fontSize: "12px", paddingRight: "10px", paddingTop: "10px" }}>
            © 2024 Pragyan Shrestha
          </Typography>
      </AppBar>
    </div>
  );
};
export default App
