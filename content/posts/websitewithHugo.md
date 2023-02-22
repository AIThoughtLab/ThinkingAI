+++
title = "How to create a static website using hugo? "
description = "Hugo, the world's fastest framework for building websites"
date = "2019-02-28"
aliases = ["about-us", "about-hugo", "contact"]
author = "AGM"
+++



Creating a website using Hugo is a straightforward process that can be broken down into a few simple steps. 
Why hugo?

**Pros**                                 
* No server side code                    
* Fast to render
* Often more secure
* Content is versioned

**Cons**
* No dynamic contents
* No database
* No real time UI etc


Let us try out now; 

1. **Install Hugo**: Before you can start creating your website, you'll need to install Hugo on your computer. You can download the latest version of Hugo from the Hugo website (https://gohugo.io/getting-started/installing/). If you wish to use rich contents in your website, make sure to download the hugo extended. 


2. **Create a new site**: Once you have Hugo installed, you can create a new site by running the following command in your terminal: 

     **hugo new site mysite** (replace "mysite" with the name of your site)
     
3. **Choose a theme**: Hugo has a wide variety of themes that you can choose from, you can either find a theme from the official theme library (https://themes.gohugo.io/) or you can create your own theme. Once you've chosen a theme, add it to your site by copying the theme files into the "themes" directory of your site. We will create our website using **hugo-coder**. Follow the steps;

* Initialize the git using **git init**.
* Run the following command to download the **hugo-coder** theme in the theme directory using **submodule** command; 

  > git submodule add https://github.com/luizdepra/hugo-coder themes/hugo-coder
  
4. Now let us copy the **exampleSite**, (which is inside the theme) to the main website folder using the following command;
  
  > cp themes/hugo-coder/exampleSite/* -rf ~/Documents/myGitHub/ThinkingAI/
  
5. Create content: Once you have a theme set up, you can start creating content for your site. You can create new content by running the following command in your terminal: **hugo new post/my-first-post.md** (replace "my-first-post" with the name of your post)

   Note: If you want to deploy the website in the Github page, delete the draft or make it false. 
   
   See the figure below;
   
   ![alt text](/images/Draft.png)
   
   
6. Set baseURL = "https://AIThoughtLab.github.io/ThinkingAI/" in the **config.toml**

7. Build your site: Once you have created your content, you can build your site by running the following command in your terminal: **hugo**

8. Serve your site (local testing): After building your site, you can serve it locally by running the following command in your terminal: **hugo server -D** or **hugo server**

9. Deploy your site: Now create a folder name - **.github** in your website folder, navigate into it and create **workflows** folder.

	Create a **gh-pages.yml** page and paste the following code;
	Structure: **.github/workflows/gh-pages.yml**
	
	You can find the gh-pages code from the following link;
        https://gohugo.io/hosting-and-deployment/hosting-on-github/
        
10. Now go to your Github page and create a new repository; e.g. **blog** (make it public)

    In the main page of your Github, go to the **Settings/Developer settings** and generate **Personal access tokens**. This is important when push the website into your repository. Under the **select scopes**, tick **repo, workflow, write:packages, admin:org, admin:repo_hook, delete_repo**
    
    
    Simillary; go the **Settings** of the repository and go to **Actions** > **General**.. and then under the **Workflow permissions**, choose **Read and Write Permissions**
    
11. Finally run the following commands;

Check **git status**. If there files to be commited add it. 
   * **git add .**
   * **git commit -m "first commit"**
   * **git branch -M main**
   * **git remote add origin https://github.com/AIThoughtLab/blog.git**
   * **git push -u origin main**


Finally go to **Setting > Pages** and then go to Branch and select **gh-pages**

![alt text](/images/gh.png)

These are the basic steps for creating a website using Hugo. Depending on your specific needs and the theme you choose, there may be additional steps or customization options available.




   


   
   
   












