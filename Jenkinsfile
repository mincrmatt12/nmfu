pipeline {
	agent {
		docker { image 'python:3' }
	}
	stages {
		stage ('Build') {
			steps {
				sh "python setup.py egg_info --tag-build 'dev0+git-${env.GIT_COMMIT}' sdist bdist_wheel"
				archiveArtifacts artifacts: 'dist/*'
			}
		}
	}
}
